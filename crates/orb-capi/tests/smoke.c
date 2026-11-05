/* The C ABI's smoke test (plan §14.1): the installed headers and library
 * link, a map draws a style into a texture target, and errors carry a
 * status and a message. The full binding scenario runs in Rust
 * (`orb-api`'s tests); the emitters' tests cover every API shape.
 * `cargo xtask bindings --test` builds it against the installed library. */

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

int main(void) {
    CHECK(OrbRuntime_abi_version() == 3);
    OrbRuntime_create_result rt = OrbRuntime_create(0);
    CHECK(rt.is_ok);
    OrbTarget* target = OrbTarget_create_texture(rt.ok, SIZE, SIZE, 1.0);
    OrbMap* map = OrbMap_create(rt.ok, OrbFeel_Desktop, 0, 0, 0, 0);

    OrbMap_set_style_json_result bad = OrbMap_set_style_json(map, view("{not json"));
    CHECK(!bad.is_ok && bad.err == OrbError_InvalidArgument);
    DiplomatWrite* message = diplomat_buffer_write_create(0);
    OrbError_last_message(message);
    CHECK(diplomat_buffer_write_len(message) > 0);
    diplomat_buffer_write_destroy(message);

    /* Draws until the blue box shows in the middle. */
    CHECK(OrbMap_set_style_json(map, view(STYLE)).is_ok);
    time_t start = time(NULL);
    for (;;) {
        CHECK(OrbMap_render(map, target).is_ok);
        CHECK(OrbTarget_read_pixels(target).is_ok);
        DiplomatU8View pixels = OrbTarget_pixels(target);
        CHECK(pixels.len == SIZE * SIZE * 4);
        const unsigned char* c = pixels.data + (SIZE / 2 * SIZE + SIZE / 2) * 4;
        if (c[0] == 0 && c[1] == 0 && c[2] == 255 && c[3] == 255) break;
        CHECK(time(NULL) - start < 30);
        struct timespec ms = {0, 2000000};
        nanosleep(&ms, NULL);
    }

    OrbFeature* hit = OrbMap_hit_test(map, SIZE / 2.0, SIZE / 2.0);
    CHECK(hit && OrbFeature_has_id(hit));
    OrbFeature_destroy(hit);

    OrbMap_destroy(map);
    OrbTarget_destroy(target);
    OrbRuntime_destroy(rt.ok);
    printf("smoke.c: ok\n");
    return 0;
}
