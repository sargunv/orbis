#define _GNU_SOURCE /* memmem */
/* The binding scenario through the C ABI (plan §14.1): a style into a
 * texture target, a hit test, an edit, a style image, camera moves, a
 * click, events, idle and errors.
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

/* Every pixel of an 8×8 image one colour. */
static void fill(unsigned char image[8 * 8 * 4], unsigned char r, unsigned char g, unsigned char b) {
    for (int i = 0; i < 8 * 8; i++) {
        image[i * 4] = r;
        image[i * 4 + 1] = g;
        image[i * 4 + 2] = b;
        image[i * 4 + 3] = 255;
    }
}

static void pixel(OrbTarget* target, int x, int y, unsigned char rgba[4]) {
    CHECK(OrbTarget_read_pixels(target).is_ok);
    DiplomatU8View pixels = OrbTarget_pixels(target);
    CHECK(pixels.len == SIZE * SIZE * 4);
    memcpy(rgba, pixels.data + (y * SIZE + x) * 4, 4);
}

/* Draws until a frame is idle and the map needs no more; counts the events
 * of each kind. Between frames a map waiting for its jobs needs none: a
 * host would wait for its wake, and this polls. */
static void settle(OrbMap* map, OrbTarget* target, int counts[16]) {
    time_t start = time(NULL);
    int idle = 0;
    while (!idle || OrbMap_needs_frame(map)) {
        OrbMap_render_result r = OrbMap_render(map, target);
        CHECK(r.is_ok);
        idle = r.ok.drawn && r.ok.idle;
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
    CHECK(OrbRuntime_abi_version() == 3);
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

    /* Rendered features arrive as an event with the query's id. */
    uint64_t query = OrbMap_query_rendered(map, SIZE / 2.0, SIZE / 2.0, SIZE / 2.0, SIZE / 2.0);
    int answered = 0;
    for (int tries = 0; tries < 5000 && !answered; tries++) {
        OrbEvent* q;
        while ((q = OrbMap_poll_event(map))) {
            if (OrbEvent_kind(q) == OrbEventKind_QueryResult && OrbEvent_id(q) == query) {
                CHECK(OrbEvent_flag(q));
                DiplomatWrite* json = diplomat_buffer_write_create(0);
                OrbEvent_text(q, json);
                char* bytes = diplomat_buffer_write_get_bytes(json);
                size_t len = diplomat_buffer_write_len(json);
                CHECK(len > 2 && bytes[0] == '[');
                const char* needle = "\"id\":\"box\"";
                CHECK(memmem(bytes, len, needle, strlen(needle)) != NULL);
                diplomat_buffer_write_destroy(json);
                answered = 1;
            }
            OrbEvent_destroy(q);
        }
        struct timespec ms = {0, 1000000};
        nanosleep(&ms, NULL);
    }
    CHECK(answered);
    settle(map, target, counts);

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

    /* A pattern the style lacks is reported; added, it draws, and new
     * pixels replace it in place. */
    CHECK(OrbMap_edit(map, view("[[\"setPaintProperty\", \"box\", \"fill-pattern\", \"dot\"]]")).is_ok);
    memset(counts, 0, sizeof counts);
    settle(map, target, counts);
    CHECK(counts[OrbEventKind_StyleImageMissing] == 1);
    CHECK(!OrbMap_has_image(map, view("dot")));
    unsigned char image[8 * 8 * 4];
    fill(image, 255, 255, 0);
    DiplomatU8View pixels = {image, sizeof image};
    CHECK(OrbMap_add_image(map, view("dot"), 8, 8, pixels, view("{\"pixelRatio\": 1}")).is_ok);
    CHECK(OrbMap_has_image(map, view("dot")));
    OrbMap_add_image_result again = OrbMap_add_image(map, view("dot"), 8, 8, pixels, view(""));
    CHECK(!again.is_ok && again.err == OrbError_InvalidArgument);
    settle(map, target, counts);
    pixel(target, SIZE / 2, SIZE / 2, rgba);
    CHECK(rgba[0] == 255 && rgba[1] == 255 && rgba[2] == 0);
    fill(image, 255, 0, 255);
    CHECK(OrbMap_update_image(map, view("dot"), 8, 8, pixels).is_ok);
    settle(map, target, counts);
    pixel(target, SIZE / 2, SIZE / 2, rgba);
    CHECK(rgba[0] == 255 && rgba[1] == 0 && rgba[2] == 255);
    CHECK(OrbMap_remove_image(map, view("dot")).is_ok);
    CHECK(!OrbMap_has_image(map, view("dot")));
    CHECK(!OrbMap_remove_image(map, view("dot")).is_ok);
    CHECK(OrbMap_edit(map, view("[[\"setPaintProperty\", \"box\", \"fill-pattern\", null]]")).is_ok);
    settle(map, target, counts);

    /* Camera moves carry a roll and an elevation; unclamped, the centre
     * keeps the elevation it is given. */
    OrbCameraTarget rolled = {NAN, NAN, NAN, NAN, NAN, 30, NAN};
    OrbMap_jump_to(map, rolled);
    CHECK(OrbMap_camera(map).roll == 30);
    OrbMap_set_center_clamped_to_ground(map, false);
    CHECK(!OrbMap_center_clamped_to_ground(map));
    OrbMap_set_center_elevation(map, 120);
    CHECK(OrbMap_camera(map).elevation == 120);
    OrbCameraTarget level = {NAN, NAN, NAN, NAN, NAN, 0, 0};
    OrbMap_ease_to(map, level, 50);
    settle(map, target, counts);
    camera = OrbMap_camera(map);
    CHECK(camera.roll == 0 && camera.elevation == 0);
    OrbMap_set_center_clamped_to_ground(map, true);

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
