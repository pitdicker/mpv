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
#include <libplacebo/utils/frame_queue.h>

#include "config.h"
#include "common/common.h"
#include "video/mp_image.h"
#include "video/fmt-conversion.h"
#include "gpu/context.h"
#include "sub/osd.h"

#if HAVE_VULKAN
#include "vulkan/context.h"
#endif

enum preset {
    PRESET_DEFAULT = 0,
    PRESET_LOW = 1,
    PRESET_HIGH = 2,
};

static const struct pl_render_params low_quality_params = {0};
static const struct pl_render_params *presets[] = {
    [PRESET_DEFAULT] = &pl_render_default_params,
    [PRESET_LOW]     = &low_quality_params,
    [PRESET_HIGH]    = &pl_render_high_quality_params,
};

struct osd_entry {
    const struct pl_tex *tex;
    struct pl_overlay_part *parts;
    int num_parts;
};

struct osd_state {
    struct osd_entry entries[MAX_OSD_PARTS];
    struct pl_overlay overlays[MAX_OSD_PARTS];
};

struct priv {
    struct mp_log *log;
    struct ra_ctx *ra_ctx;
    struct ra_ctx_opts opts;
    int preset;

    struct pl_context *ctx;
    struct pl_renderer *rr;
    struct pl_queue *queue;
    const struct pl_gpu *gpu;
    const struct pl_swapchain *sw;
    const struct pl_fmt *osd_fmt[SUBBITMAP_COUNT];
    const struct pl_tex **sub_tex;
    int num_sub_tex;

    struct mp_rect src, dst;
    struct mp_osd_res osd_res;
    struct osd_state osd_state;
    struct bstr icc_profile;
    uint64_t icc_signature;

    uint64_t last_id;
    double last_src_pts;
    double last_dst_pts;
    bool is_interpolated;
};

static void reset_queue(struct priv *p)
{
    pl_queue_reset(p->queue);
    p->last_id = 0;
    p->last_src_pts = 0.0;
    p->last_dst_pts = 0.0;
}

static void write_overlays(struct vo *vo, struct mp_osd_res res, double pts,
                           int flags, struct osd_state *state,
                           struct pl_frame *frame)
{
    struct priv *p = vo->priv;
    static const bool subfmt_all[SUBBITMAP_COUNT] = {
        [SUBBITMAP_LIBASS] = true,
        [SUBBITMAP_RGBA]   = true,
    };

    struct sub_bitmap_list *subs = osd_render(vo->osd, res, pts, flags, subfmt_all);
    frame->num_overlays = 0;
    frame->overlays = state->overlays;

    for (int n = 0; n < subs->num_items; n++) {
        const struct sub_bitmaps *item = subs->items[n];
        if (!item->num_parts || !item->packed)
            continue;
        struct osd_entry *entry = &state->entries[item->render_index];
        const struct pl_fmt *tex_fmt = p->osd_fmt[item->format];
        MP_TARRAY_POP(p->sub_tex, p->num_sub_tex, &entry->tex);
        bool ok = pl_tex_recreate(p->gpu, &entry->tex, &(struct pl_tex_params) {
            .format = tex_fmt,
            .w = MPMAX(item->packed_w, entry->tex ? entry->tex->params.w : 0),
            .h = MPMAX(item->packed_h, entry->tex ? entry->tex->params.h : 0),
            .host_writable = true,
            .sampleable = true,
        });
        if (!ok) {
            MP_ERR(vo, "Failed recreating OSD texture!\n");
            break;
        }
        ok = pl_tex_upload(p->gpu, &(struct pl_tex_transfer_params) {
            .tex        = entry->tex,
            .rc         = { .x1 = item->packed_w, .y1 = item->packed_h, },
            .stride_w   = item->packed->stride[0] / tex_fmt->texel_size,
            .ptr        = item->packed->planes[0],
        });
        if (!ok) {
            MP_ERR(vo, "Failed uploading OSD texture!\n");
            break;
        }

        entry->num_parts = 0;
        for (int i = 0; i < item->num_parts; i++) {
            const struct sub_bitmap *b = &item->parts[i];
            uint32_t c = b->libass.color;
            MP_TARRAY_APPEND(p, entry->parts, entry->num_parts, (struct pl_overlay_part) {
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

        struct pl_overlay *ol = &state->overlays[frame->num_overlays++];
        *ol = (struct pl_overlay) {
            .tex = entry->tex,
            .parts = entry->parts,
            .num_parts = entry->num_parts,
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

    talloc_free(subs);
}

struct frame_priv {
    struct vo *vo;
    struct osd_state subs;
};

static bool map_frame(const struct pl_gpu *gpu, const struct pl_tex **tex,
                      const struct pl_source_frame *src, struct pl_frame *out_frame)
{
    struct mp_image *mpi = src->frame_data;
    struct frame_priv *fp = mpi->priv;
    struct vo *vo = fp->vo;

    struct AVFrame *avframe = mp_image_to_av_frame(mpi);
    bool ok = pl_upload_avframe(gpu, out_frame, tex, avframe);
    av_frame_free(&avframe);
    if (!ok) {
        MP_ERR(vo, "Failed uploading frame!\n");
        return false;
    }

    // Generate subtitles for this frame
    struct mp_osd_res vidres = { mpi->w, mpi->h };
    write_overlays(vo, vidres, mpi->pts, OSD_DRAW_SUB_ONLY, &fp->subs, out_frame);
    return true;
}

static void unmap_frame(const struct pl_gpu *gpu, struct pl_frame *frame,
                        const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->vo->priv;
    for (int i = 0; i < MP_ARRAY_SIZE(fp->subs.entries); i++) {
        const struct pl_tex *tex = fp->subs.entries[i].tex;
        if (tex)
            MP_TARRAY_APPEND(p, p->sub_tex, p->num_sub_tex, tex);
    }
    talloc_free(mpi);
}

static void discard_frame(const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    talloc_free(mpi);
}

static void draw_frame(struct vo *vo, struct vo_frame *frame)
{
    struct priv *p = vo->priv;
    const struct pl_gpu *gpu = p->gpu;

    // Push all incoming frames into the frame queue
    for (int n = 0; n < frame->num_frames; n++) {
        int id = frame->frame_id + n;
        if (id <= p->last_id)
            continue; // don't re-upload already seen frames

        struct mp_image *mpi = mp_image_new_ref(frame->frames[n]);
        struct frame_priv *fp = talloc_zero(mpi, struct frame_priv);
        mpi->priv = fp;
        fp->vo = vo;

        // mpv sometimes glitches out and sends frames with backwards PTS
        // discontinuities, this safeguard makes sure we always handle it
        if (mpi->pts < p->last_src_pts)
            reset_queue(p);

        pl_queue_push(p->queue, &(struct pl_source_frame) {
            .pts = mpi->pts,
            .frame_data = mpi,
            .map = map_frame,
            .unmap = unmap_frame,
            .discard = discard_frame,
        });

        p->last_src_pts = mpi->pts;
        p->last_id = id;
    }

    struct pl_swapchain_frame swframe;
    if (!pl_swapchain_start_frame(p->sw, &swframe))
        return;

    bool valid = false;
    p->is_interpolated = false;

    // Calculate target
    struct pl_frame target;
    pl_frame_from_swapchain(&target, &swframe);
    write_overlays(vo, p->osd_res, 0, OSD_DRAW_OSD_ONLY, &p->osd_state, &target);
    target.crop = (struct pl_rect2df) { p->dst.x0, p->dst.y0, p->dst.x1, p->dst.y1 };
    target.profile = (struct pl_icc_profile) {
        .signature = p->icc_signature,
        .data = p->icc_profile.start,
        .len = p->icc_profile.len,
    };

    struct pl_frame_mix mix = {0};
    if (frame->current) {
        // Update queue state
        struct pl_queue_params qparams = {
            .pts = frame->current->pts + frame->vsync_offset,
            .radius = pl_frame_mix_radius(presets[p->preset]),
            .vsync_duration = frame->vsync_interval,
            .frame_duration = frame->ideal_frame_duration,
        };

        // mpv likes to generate sporadically jumping PTS shortly after
        // initialization, but pl_queue does not like these. Hard-clamp as
        // a simple work-around.
        qparams.pts = MPMAX(qparams.pts, p->last_dst_pts);
        p->last_dst_pts = qparams.pts;

        switch (pl_queue_update(p->queue, &mix, &qparams)) {
        case PL_QUEUE_ERR:
            MP_ERR(vo, "Failed updating frames!\n");
            goto done;
        case PL_QUEUE_EOF:
            abort(); // we never signal EOF
        case PL_QUEUE_MORE:
        case PL_QUEUE_OK:
            break;
        }

        const struct pl_frame *still_frame;
        if (frame->still && mix.num_frames > 1) {
            for (int i = 0; i < mix.num_frames; i++) {
                if (i && mix.timestamps[i] > 0.0)
                    break;
                still_frame = mix.frames[i];
            }
            mix.frames = &still_frame;
            mix.num_frames = 1;
        }

        // Update source crop on all existing frames. We technically own the
        // `pl_frame` struct so this is kosher.
        //
        // XXX: why is this needed? how come doing it in `map_frame` isn't as
        // smooth as doing it here?
        for (int i = 0; i < mix.num_frames; i++) {
            struct pl_frame *img = (struct pl_frame *) mix.frames[i];
            img->crop = (struct pl_rect2df) {
                p->src.x0, p->src.y0, p->src.x1, p->src.y1,
            };
        }
    }

    // Render frame
    if (!pl_render_image_mix(p->rr, &mix, &target, presets[p->preset])) {
        MP_ERR(vo, "Failed rendering frame!\n");
        goto done;
    }

    p->is_interpolated = mix.num_frames > 1;
    valid = true;
    // fall through

done:
    if (!valid) // clear with purple to indicate error
        pl_tex_clear(gpu, swframe.fbo, (float[4]){ 0.5, 0.0, 1.0, 1.0 });

    if (!pl_swapchain_submit_frame(p->sw))
        MP_ERR(vo, "Failed presenting frame!\n");
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
    vo_get_src_dst_rects(vo, &p->src, &p->dst, &p->osd_res);
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
    case VOCTRL_PAUSE:
        if (p->is_interpolated)
            vo->want_redraw = true;
        return VO_TRUE;

    case VOCTRL_RESET:
        pl_renderer_flush_cache(p->rr);
        reset_queue(p);
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
    for (int i = 0; i < MP_ARRAY_SIZE(p->osd_state.entries); i++)
        pl_tex_destroy(p->gpu, &p->osd_state.entries[i].tex);
    for (int i = 0; i < p->num_sub_tex; i++)
        pl_tex_destroy(p->gpu, &p->sub_tex[i]);
    pl_queue_destroy(&p->queue);
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
    p->queue = pl_queue_create(p->gpu);
    p->osd_fmt[SUBBITMAP_LIBASS] = pl_find_named_fmt(p->gpu, "r8");
    p->osd_fmt[SUBBITMAP_RGBA] = pl_find_named_fmt(p->gpu, "rgba8");

    // TODO: pl_renderer_save/load

    // Request as many frames as possible from the decoder. This is not really
    // wasteful since we pass these through libplacebo's frame queueing
    // mechanism, which only uploads frames on an as-needed basis.
    vo_set_queue_params(vo, 0, VO_MAX_REQ_FRAMES);
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
    .preinit = preinit,
    .query_format = query_format,
    .reconfig = reconfig,
    .control = control,
    //.get_image = get_image,
    .draw_frame = draw_frame,
    .flip_page = flip_page,
    .get_vsync = get_vsync,
    .wait_events = wait_events,
    .wakeup = wakeup,
    .uninit = uninit,
    .priv_size = sizeof(struct priv),
    .options = options,
};
