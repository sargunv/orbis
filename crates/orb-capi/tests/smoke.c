/* The binding scenario through the C ABI (plan §14.1): a style into a
 * texture target, a hit test, an edit, a click, events, idle and errors.
 * `cargo xtask capi --test` builds it against the installed library. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <orbis.h>

#define SIZE 64

static const char STYLE[] =
    "{\"version\": 8, \"center\": [0, 0], \"zoom\": 0,"
    " \"sources\": {\"box\": {\"type\": \"geojson\", \"data\": {"
    "  \"type\": \"Feature\", \"id\": 7, \"properties\": {},"
    "  \"geometry\": {\"type\": \"Polygon\", \"coordinates\":"
    "   [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]]}}}},"
    " \"layers\": ["
    "  {\"id\": \"background\", \"type\": \"background\", \"paint\": {\"background-color\": \"#ff0000\"}},"
    "  {\"id\": \"box\", \"type\": \"fill\", \"source\": \"box\", \"paint\": {\"fill-color\": \"#0000ff\"}}]}";

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "smoke.c:%d: check failed: %s\n", __LINE__, #cond); \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

static DiplomatStringView view(const char* text) {
    DiplomatStringView v = {text, strlen(text)};
    return v;
}

/* Text through a DiplomatWrite, NUL-terminated into `out`. */
static void text_of(void (*f)(const OrbFeature*, DiplomatWrite*), const OrbFeature* feature,
                    char* out, size_t cap) {
    DiplomatWrite* w = diplomat_buffer_write_create(0);
    f(feature, w);
    size_t len = diplomat_buffer_write_len(w);
    if (len >= cap) len = cap - 1;
    memcpy(out, diplomat_buffer_write_get_bytes(w), len);
    out[len] = 0;
    diplomat_buffer_write_destroy(w);
}

static void pixel(OrbTarget* target, int x, int y, unsigned char rgba[4]) {
    CHECK(OrbTarget_read_pixels(target).is_ok);
    DiplomatU8View pixels = OrbTarget_pixels(target);
    CHECK(pixels.len == SIZE * SIZE * 4);
    memcpy(rgba, pixels.data + (y * SIZE + x) * 4, 4);
}

/* Draws until the map needs no frames; counts the events of each kind. */
static void settle(OrbMap* map, OrbTarget* target, int counts[16]) {
    time_t start = time(NULL);
    while (OrbMap_needs_frame(map)) {
        OrbMap_render_result r = OrbMap_render(map, target);
        CHECK(r.is_ok);
        CHECK(time(NULL) - start < 30);
        struct timespec ms = {0, 2000000};
        nanosleep(&ms, NULL);
    }
    OrbEvent* e;
    while ((e = OrbMap_poll_event(map))) {
        counts[OrbEvent_kind(e)]++;
        OrbEvent_destroy(e);
    }
}

int main(void) {
    CHECK(OrbRuntime_abi_version() == 1);
    OrbRuntime_create_result rt = OrbRuntime_create(0);
    CHECK(rt.is_ok);
    OrbTarget* target = OrbTarget_create_texture(rt.ok, SIZE, SIZE, 1.0);
    OrbMap* map = OrbMap_create(rt.ok, OrbFeel_Desktop, 0, 0, 0, 0);

    /* Errors carry a status and a message. */
    OrbMap_set_style_json_result bad = OrbMap_set_style_json(map, view("{not json"));
    CHECK(!bad.is_ok && bad.err == OrbError_InvalidArgument);
    DiplomatWrite* message = diplomat_buffer_write_create(0);
    OrbError_last_message(message);
    CHECK(diplomat_buffer_write_len(message) > 0);
    diplomat_buffer_write_destroy(message);
    OrbMap_edit_result early = OrbMap_edit(map, view("[]"));
    CHECK(!early.is_ok && early.err == OrbError_InvalidState);

    CHECK(OrbMap_set_style_json(map, view(STYLE)).is_ok);
    int counts[16] = {0};
    settle(map, target, counts);
    CHECK(counts[OrbEventKind_StyleLoad] == 1);
    CHECK(counts[OrbEventKind_Load] == 1);
    CHECK(counts[OrbEventKind_Idle] == 1);
    OrbCamera camera = OrbMap_camera(map);
    CHECK(fabs(camera.zoom) < 1e-9);

    unsigned char rgba[4];
    pixel(target, SIZE / 2, SIZE / 2, rgba);
    CHECK(rgba[0] == 0 && rgba[1] == 0 && rgba[2] == 255 && rgba[3] == 255);
    pixel(target, 1, 1, rgba);
    CHECK(rgba[0] == 255 && rgba[2] == 0);

    OrbFeature* hit = OrbMap_hit_test(map, SIZE / 2.0, SIZE / 2.0);
    CHECK(hit);
    char text[64];
    text_of(OrbFeature_layer, hit, text, sizeof text);
    CHECK(strcmp(text, "box") == 0);
    CHECK(OrbFeature_has_id(hit));
    text_of(OrbFeature_id, hit, text, sizeof text);
    CHECK(strcmp(text, "7") == 0);
    OrbFeature_destroy(hit);
    CHECK(!OrbMap_hit_test(map, 1, 1));

    OrbMap_edit_result edit =
        OrbMap_edit(map, view("[[\"setPaintProperty\", \"box\", \"fill-color\", \"#00ff00\"]]"));
    CHECK(edit.is_ok);
    uint64_t shown = 0;
    while (OrbMap_needs_frame(map)) {
        OrbMap_render_result r = OrbMap_render(map, target);
        CHECK(r.is_ok);
        if (r.ok.generation > shown) shown = r.ok.generation;
    }
    CHECK(shown >= edit.ok);
    pixel(target, SIZE / 2, SIZE / 2, rgba);
    CHECK(rgba[0] == 0 && rgba[1] == 255 && rgba[2] == 0);

    /* A click comes back as an event at its position. */
    OrbModifiers none = {false, false, false, false};
    OrbMap_pointer_down(map, 1, OrbPointerKind_Mouse, OrbButton_Primary, 10, 20, none);
    OrbMap_pointer_up(map, 1, 10, 20);
    OrbMap_render(map, target);
    int clicked = 0;
    OrbEvent* e;
    while ((e = OrbMap_poll_event(map))) {
        if (OrbEvent_kind(e) == OrbEventKind_Click) {
            CHECK(OrbEvent_x(e) == 10 && OrbEvent_y(e) == 20);
            clicked = 1;
        }
        OrbEvent_destroy(e);
    }
    CHECK(clicked);

    /* project and unproject agree. */
    OrbPoint p = OrbMap_project(map, 5, 5);
    CHECK(p.valid);
    OrbLngLat back = OrbMap_unproject(map, p.x, p.y);
    CHECK(back.valid && fabs(back.lon - 5) < 1e-6 && fabs(back.lat - 5) < 1e-6);

    OrbMap_destroy(map);
    OrbTarget_destroy(target);
    OrbRuntime_destroy(rt.ok);
    printf("smoke.c: ok\n");
    return 0;
}
