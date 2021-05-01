#pragma once

#include "common/common.h"
#include "common/msg.h"
#include "video/csputils.h"

#include <libplacebo/common.h>
#include <libplacebo/colorspace.h>

void mppl_ctx_set_log(struct pl_context *ctx, struct mp_log *log, bool probing);

static inline struct pl_rect2d mp_rect2d_to_pl(struct mp_rect rc)
{
    return (struct pl_rect2d) {
        .x0 = rc.x0,
        .y0 = rc.y0,
        .x1 = rc.x1,
        .y1 = rc.y1,
    };
}

enum pl_color_primaries mp_prim_to_pl(enum mp_csp_prim prim);
enum pl_color_transfer mp_trc_to_pl(enum mp_csp_trc trc);
