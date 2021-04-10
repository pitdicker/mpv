/*
 * Copyright (C) 2021 Niklas Haas
 *
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/utils/libav.h>

#include "config.h"
#include "common/common.h"
#include "video/mp_image.h"
#include "video/fmt-conversion.h"
#include "gpu/context.h"
#include "sub/osd.h"

#if HAVE_VULKAN
#include "vulkan/context.h"
#endif

struct overlay {
    const struct pl_tex *tex;
    struct pl_overlay_part *parts;
    int num_parts;
};

static const bool subfmt_all[SUBBITMAP_COUNT] = {
    [SUBBITMAP_LIBASS] = true,
    [SUBBITMAP_RGBA]   = true,
};

enum preset {
    PRESET_DEFAULT = 0,
    PRESET_LOW = 1,
    PRESET_HIGH = 2,
};

static const struct pl_render_params low_quality_params = {0};

struct priv {
    struct mp_log *log;
    struct ra_ctx *ra_ctx;
    struct ra_ctx_opts opts;
    int preset;

    struct pl_context *ctx;
    struct pl_renderer *rr;
    const struct pl_gpu *gpu;
    const struct pl_swapchain *sw;
    const struct pl_tex *video_tex[4];
    const struct pl_fmt *osd_fmt[SUBBITMAP_COUNT];
    struct overlay osd[MAX_OSD_PARTS];

    struct mp_rect src, dst;
    struct mp_osd_res osdres;
    struct bstr icc_profile;
    uint64_t icc_signature;
};

static void write_overlays(struct vo *vo, struct pl_frame *frame,
                           const struct sub_bitmap_list *subs)
{
    struct priv *p = vo->priv;

    for (int n = 0; n < subs->num_items; n++) {
        const struct sub_bitmaps *item = subs->items[n];
        if (!item->num_parts || !item->packed)
            continue;
        struct overlay *osd = &p->osd[item->render_index];
        const struct pl_fmt *tex_fmt = p->osd_fmt[item->format];
        bool ok = pl_tex_recreate(p->gpu, &osd->tex, &(struct pl_tex_params) {
            .format = tex_fmt,
            .w = MPMAX(item->packed_w, osd->tex ? osd->tex->params.w : 0),
            .h = MPMAX(item->packed_h, osd->tex ? osd->tex->params.h : 0),
            .host_writable = true,
            .sampleable = true,
        });
        if (!ok) {
            MP_ERR(vo, "Failed recreating OSD texture!\n");
            break;
        }
        ok = pl_tex_upload(p->gpu, &(struct pl_tex_transfer_params) {
            .tex        = osd->tex,
            .rc         = { .x1 = item->packed_w, .y1 = item->packed_h, },
            .stride_w   = item->packed->stride[0] / tex_fmt->texel_size,
            .ptr        = item->packed->planes[0],
        });
        if (!ok) {
            MP_ERR(vo, "Failed uploading OSD texture!\n");
            break;
        }

        osd->num_parts = 0;
        for (int i = 0; i < item->num_parts; i++) {
            const struct sub_bitmap *b = &item->parts[i];
            uint32_t c = b->libass.color;
            MP_TARRAY_APPEND(vo, osd->parts, osd->num_parts, (struct pl_overlay_part) {
                .src = { b->src_x, b->src_y, b->src_x + b->w, b->src_y + b->h },
                .dst = { b->x, b->y, b->x + b->dw, b->y + b->dh },
                .color = {
                    (c >> 24) / 255.0,
                    ((c >> 16) & 0xFF) / 255.0,
                    ((c >> 8) & 0xFF) / 255.0,
                    1.0 - (c & 0xFF) / 255.0,
                }
            });
        }

        int ol_idx = frame->num_overlays++;
        struct pl_overlay *ol = (struct pl_overlay *) &frame->overlays[ol_idx];
        *ol = (struct pl_overlay) {
            .tex = osd->tex,
            .parts = osd->parts,
            .num_parts = osd->num_parts,
            .color = frame->color,
        };

        switch (item->format) {
        case SUBBITMAP_RGBA:
            ol->mode = PL_OVERLAY_NORMAL;
            ol->repr.alpha = PL_ALPHA_PREMULTIPLIED;
            break;
        case SUBBITMAP_LIBASS:
            ol->mode = PL_OVERLAY_MONOCHROME;
            ol->repr.alpha = PL_ALPHA_INDEPENDENT;
            break;
        }
    }
}

static void draw_image(struct vo *vo, mp_image_t *mpi)
{
    struct priv *p = vo->priv;
    const struct pl_gpu *gpu = p->gpu;
    struct pl_swapchain_frame swframe;
    if (!pl_swapchain_start_frame(p->sw, &swframe)) {
        talloc_free(mpi);
        return;
    }

    bool valid = false;

    // Calculate target
    struct pl_frame target;
    pl_frame_from_swapchain(&target, &swframe);
    target.crop = (struct pl_rect2df) { p->dst.x0, p->dst.y0, p->dst.x1, p->dst.y1 };
    target.profile = (struct pl_icc_profile) {
        .signature = p->icc_signature,
        .data = p->icc_profile.start,
        .len = p->icc_profile.len,
    };

    // Calculate source frame
    struct pl_frame image;
    struct AVFrame *avframe = mp_image_to_av_frame(mpi);
    if (!pl_upload_avframe(gpu, &image, p->video_tex, avframe)) {
        MP_ERR(vo, "Failed uploading frame!\n");
        goto error;
    }
    image.crop  = (struct pl_rect2df) { p->src.x0, p->src.y0, p->src.x1, p->src.y1 };

    // Update overlays
    struct pl_overlay image_ol[MAX_OSD_PARTS], target_ol[MAX_OSD_PARTS];
    struct sub_bitmap_list *osd, *sub;
    struct mp_osd_res vidres = { mpi->w, mpi->h };
    osd = osd_render(vo->osd, p->osdres, mpi->pts, OSD_DRAW_OSD_ONLY, subfmt_all);
    sub = osd_render(vo->osd, vidres, mpi->pts, OSD_DRAW_SUB_ONLY, subfmt_all);
    target.overlays = target_ol;
    image.overlays = image_ol;
    write_overlays(vo, &target, osd);
    write_overlays(vo, &image, sub);
    talloc_free(osd);
    talloc_free(sub);

    // Render frame
    if (pl_frame_is_cropped(&target))
        pl_tex_clear(gpu, swframe.fbo, (float[4]){ 0.0, 0.0, 0.0, 1.0 });

    static const struct pl_render_params *presets[] = {
        [PRESET_DEFAULT] = &pl_render_default_params,
        [PRESET_LOW]     = &low_quality_params,
        [PRESET_HIGH]    = &pl_render_high_quality_params,
    };

    if (!pl_render_image(p->rr, &image, &target, presets[p->preset])) {
        MP_ERR(vo, "Failed rendering frame!\n");
        goto error;
    }

    valid = true;
    // fall through

error:
    if (!valid) // clear with purple to indicate error
        pl_tex_clear(gpu, swframe.fbo, (float[4]){ 0.5, 0.0, 1.0, 1.0 });

    if (!pl_swapchain_submit_frame(p->sw))
        MP_ERR(vo, "Failed presenting frame!\n");

    av_frame_free(&avframe);
    talloc_free(mpi);
}

static void flip_page(struct vo *vo)
{
    struct priv *p = vo->priv;
    struct ra_swapchain *sw = p->ra_ctx->swapchain;
    sw->fns->swap_buffers(sw);
}

static void get_vsync(struct vo *vo, struct vo_vsync_info *info)
{
    struct priv *p = vo->priv;
    struct ra_swapchain *sw = p->ra_ctx->swapchain;
    if (sw->fns->get_vsync)
        sw->fns->get_vsync(sw, info);
}

static int query_format(struct vo *vo, int format)
{
    struct priv *p = vo->priv;
    enum AVPixelFormat pixfmt = imgfmt2pixfmt(format);
    return pixfmt >= 0 && pl_test_pixfmt(p->gpu, pixfmt);
}

static void resize(struct vo *vo)
{
    struct priv *p = vo->priv;
    vo_get_src_dst_rects(vo, &p->src, &p->dst, &p->osdres);
}

static int reconfig(struct vo *vo, struct mp_image_params *params)
{
    struct priv *p = vo->priv;
    if (!p->ra_ctx->fns->reconfig(p->ra_ctx))
        return -1;

    resize(vo);
    return 0;
}

static void get_and_update_icc_profile(struct priv *p, int *events)
{
    MP_VERBOSE(p, "Querying ICC profile...\n");
    bstr icc = bstr0(NULL);
    int r = p->ra_ctx->fns->control(p->ra_ctx, events, VOCTRL_GET_ICC_PROFILE, &icc);

    if (r != VO_NOTAVAIL) {
        if (r == VO_FALSE) {
            MP_WARN(p, "Could not retrieve an ICC profile.\n");
        } else if (r == VO_NOTIMPL) {
            MP_ERR(p, "icc-profile-auto not implemented on this platform.\n");
        }

        p->icc_profile = icc;
        p->icc_signature++;
    }
}

static int control(struct vo *vo, uint32_t request, void *data)
{
    struct priv *p = vo->priv;

    switch (request) {
    case VOCTRL_SET_PANSCAN:
        resize(vo);
        // fall through
    case VOCTRL_SET_EQUALIZER:
    case VOCTRL_UPDATE_RENDER_OPTS:
    case VOCTRL_RESET:
    case VOCTRL_PAUSE:
        vo->want_redraw = true;
        return VO_TRUE;
    }

    int events = 0;
    int r = p->ra_ctx->fns->control(p->ra_ctx, &events, request, data);
    if (events & VO_EVENT_ICC_PROFILE_CHANGED) {
        get_and_update_icc_profile(p, &events);
        vo->want_redraw = true;
    }
    if (events & VO_EVENT_RESIZE)
        resize(vo);
    if (events & VO_EVENT_EXPOSE)
        vo->want_redraw = true;
    vo_event(vo, events);

    return r;
}

static void wakeup(struct vo *vo)
{
    struct priv *p = vo->priv;
    if (p->ra_ctx && p->ra_ctx->fns->wakeup)
        p->ra_ctx->fns->wakeup(p->ra_ctx);
}

static void wait_events(struct vo *vo, int64_t until_time_us)
{
    struct priv *p = vo->priv;
    if (p->ra_ctx && p->ra_ctx->fns->wait_events) {
        p->ra_ctx->fns->wait_events(p->ra_ctx, until_time_us);
    } else {
        vo_wait_default(vo, until_time_us);
    }
}

static void uninit(struct vo *vo)
{
    struct priv *p = vo->priv;
    for (int i = 0; i < MP_ARRAY_SIZE(p->video_tex); i++)
        pl_tex_destroy(p->gpu, &p->video_tex[i]);
    for (int i = 0; i < MP_ARRAY_SIZE(p->osd); i++)
        pl_tex_destroy(p->gpu, &p->osd[i].tex);
    pl_renderer_destroy(&p->rr);
    ra_ctx_destroy(&p->ra_ctx);
}

static int preinit(struct vo *vo)
{
    struct priv *p = vo->priv;
    p->log = vo->log;

    p->ra_ctx = ra_ctx_create(vo, "vulkan", NULL, p->opts);
    if (!p->ra_ctx)
        goto err_out;

#if HAVE_VULKAN
    struct mpvk_ctx *vkctx = ra_vk_ctx_get(p->ra_ctx);
    if (vkctx) {
        p->ctx = vkctx->ctx;
        p->gpu = vkctx->gpu;
        p->sw = vkctx->swapchain;
        goto done;
    }
#endif

    // TODO: wrap GL contexts

    goto err_out;

done:
    p->rr = pl_renderer_create(p->ctx, p->gpu);
    p->osd_fmt[SUBBITMAP_LIBASS] = pl_find_named_fmt(p->gpu, "r8");
    p->osd_fmt[SUBBITMAP_RGBA] = pl_find_named_fmt(p->gpu, "rgba8");
    // TODO: pl_renderer_save/load
    return 0;

err_out:
    uninit(vo);
    return -1;
}

#define OPT_BASE_STRUCT struct priv
static const m_option_t options[] = {
    // FIXME: lift the gpu-* options to be global
    {"placebo-preset", OPT_CHOICE(preset,
        {"default", PRESET_DEFAULT},
        {"high",    PRESET_HIGH},
        {"low",     PRESET_LOW})},
    {"placebo-debug", OPT_FLAG(opts.debug)},
    {"placebo-sw", OPT_FLAG(opts.allow_sw)},
    {0}
};

const struct vo_driver video_out_placebo = {
    .description = "Video output based on libplacebo",
    .name = "placebo",
    //.caps = VO_CAP_ROTATE90,
    .preinit = preinit,
    .query_format = query_format,
    .reconfig = reconfig,
    .control = control,
    //.get_image = get_image,
    .draw_image = draw_image,
    .flip_page = flip_page,
    .get_vsync = get_vsync,
    .wait_events = wait_events,
    .wakeup = wakeup,
    .uninit = uninit,
    .priv_size = sizeof(struct priv),
    .options = options,
};
