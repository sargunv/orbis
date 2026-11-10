/* The C ABI's smoke test: the installed headers and library
 * link, a map draws a style into a texture target, and errors carry a
 * status and a message. The full binding scenario runs in Rust
 * (`orb-api`'s tests); the emitters' tests cover every API shape.
 * `cargo xtask bindings --test` builds it against the installed library. */

#include <math.h>
#include <orbis.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SIZE 64

static const char STYLE[] =
  "{\"version\": 8, \"center\": [0, 0], \"zoom\": 0,"
  " \"sources\": {\"box\": {\"type\": \"geojson\", \"data\": {"
  "  \"type\": \"Feature\", \"id\": 7, \"properties\": {},"
  "  \"geometry\": {\"type\": \"Polygon\", \"coordinates\":"
  "   [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]]}}}},"
  " \"layers\": ["
  "  {\"id\": \"background\", \"type\": \"background\", \"paint\": "
  "{\"background-color\": \"#ff0000\"}},"
  "  {\"id\": \"box\", \"type\": \"fill\", \"source\": \"box\", \"paint\": "
  "{\"fill-color\": \"#0000ff\"}}]}";

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      fprintf(stderr, "smoke.c:%d: check failed: %s\n", __LINE__, #cond); \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

static DiplomatStringView view(const char* text) {
  DiplomatStringView v = {text, strlen(text)};
  return v;
}

int main(void) {
  CHECK(OrbRuntime_abi_version() == 10);
  OrbRuntime_create_result rt = OrbRuntime_create(0);
  CHECK(rt.is_ok);
  OrbTarget* target = OrbTarget_create_texture(rt.ok, SIZE, SIZE, 1.0);
  OrbMap* map = OrbMap_create(rt.ok, 0, 0, 0, 0);
  /* The viewport, before a target draws. */
  OrbMap_set_viewport(map, SIZE, SIZE);
  OrbPoint middle = OrbMap_project(map, 0, 0);
  CHECK(
    middle.valid && fabs(middle.x - SIZE / 2.0) < 1e-6 &&
    fabs(middle.y - SIZE / 2.0) < 1e-6
  );

  OrbMap_set_style_json_result bad =
    OrbMap_set_style_json(map, view("{not json"));
  CHECK(!bad.is_ok && bad.err == OrbError_InvalidArgument);
  DiplomatWrite* message = diplomat_buffer_write_create(0);
  OrbError_last_message(message);
  CHECK(diplomat_buffer_write_len(message) > 0);
  diplomat_buffer_write_destroy(message);

  /* Draws until the blue box shows in the middle, from a style a request
   * rule finds on disk (files read offline). */
  char dir[] = "/tmp/orb-smoke-XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char path[256], rules[512];
  snprintf(path, sizeof path, "%s/style.json", dir);
  FILE* file = fopen(path, "w");
  CHECK(file && fputs(STYLE, file) >= 0 && fclose(file) == 0);
  snprintf(
    rules, sizeof rules, "{\"rewrites\": [[\"smoke://\", \"file://%s/\"]]}", dir
  );
  CHECK(!OrbRuntime_set_request_rules(rt.ok, view("{bad")).is_ok);
  CHECK(OrbRuntime_set_request_rules(rt.ok, view(rules)).is_ok);
  OrbRuntime_set_online(rt.ok, false);
  CHECK(!OrbRuntime_online(rt.ok));
  OrbMap_load_style_url(map, view("smoke://style.json"));
  time_t start = time(NULL);
  for (;;) {
    CHECK(OrbMap_render(map, target, 0).is_ok);
    CHECK(OrbTarget_read_pixels(target).is_ok);
    DiplomatU8View pixels = OrbTarget_pixels(target);
    CHECK(pixels.len == SIZE * SIZE * 4);
    const unsigned char* c = pixels.data + (SIZE / 2 * SIZE + SIZE / 2) * 4;
    if (c[0] == 0 && c[1] == 0 && c[2] == 255 && c[3] == 255) break;
    CHECK(time(NULL) - start < 30);
    struct timespec ms = {0, 2000000};
    nanosleep(&ms, NULL);
  }

  /* The style reads back. */
  DiplomatWrite* ids = diplomat_buffer_write_create(0);
  CHECK(OrbMap_layer_ids(map, ids).is_ok);
  CHECK(
    strncmp(
      diplomat_buffer_write_get_bytes(ids), "[\"background\",\"box\"]",
      diplomat_buffer_write_len(ids)
    ) == 0
  );
  diplomat_buffer_write_destroy(ids);
  DiplomatWrite* box = diplomat_buffer_write_create(0);
  CHECK(OrbMap_layer(map, view("box"), box).is_ok);
  CHECK(strstr(diplomat_buffer_write_get_bytes(box), "#0000ff") != NULL);
  diplomat_buffer_write_destroy(box);
  DiplomatWrite* none = diplomat_buffer_write_create(0);
  CHECK(!OrbMap_layer(map, view("nope"), none).is_ok);
  diplomat_buffer_write_destroy(none);

  OrbFeature* hit = OrbMap_hit_test(map, SIZE / 2.0, SIZE / 2.0);
  CHECK(hit && OrbFeature_has_id(hit));
  OrbFeature_destroy(hit);

  /* Edits built one by one apply as one generation; a value that isn't
   * JSON fails at once. */
  OrbTransaction* tx = OrbTransaction_create();
  CHECK(OrbTransaction_set_paint_property(
          tx, view("box"), view("fill-color"), view("\"#00ff00\"")
  )
          .is_ok);
  CHECK(OrbTransaction_set_paint_property(
          tx, view("box"), view("fill-opacity"), view("1")
  )
          .is_ok);
  CHECK(!OrbTransaction_set_filter(tx, view("box"), view("[==")).is_ok);
  uint64_t before = OrbMap_generation(map);
  OrbMap_edit_result edited = OrbMap_edit(map, tx);
  CHECK(edited.is_ok && edited.ok == before + 1);
  OrbTransaction_destroy(tx);

  /* Padding moves the centre down; a camera's projection answers before
   * any move; the eye round-trips. */
  OrbCameraTarget padded = {
    NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, {32, NAN, NAN, NAN}
  };
  OrbMap_jump_to(map, padded);
  OrbPoint centre = OrbMap_project(map, 0, 0);
  CHECK(
    centre.valid && fabs(centre.x - 32) < 1e-6 && fabs(centre.y - 48) < 1e-6
  );
  OrbCameraTarget closer = {
    NAN, NAN, 1, NAN, NAN, NAN, NAN, NAN, {NAN, NAN, NAN, NAN}
  };
  OrbPoint east = OrbMap_project_with(map, closer, 10, 0);
  CHECK(east.valid && fabs(east.x - 32 - 1024.0 * 10 / 360) < 1e-6);
  CHECK(OrbMap_camera(map).zoom == 0);
  OrbMap_jump_to_free(map, OrbMap_free_camera(map));
  OrbCamera camera = OrbMap_camera(map);
  CHECK(
    fabs(camera.lon) < 1e-9 && fabs(camera.lat) < 1e-9 &&
    fabs(camera.zoom) < 1e-9
  );
  CHECK(camera.padding.top == 32 && fabs(camera.fov - 36.87) < 0.01);

  remove(path);
  rmdir(dir);
  OrbMap_destroy(map);
  OrbTarget_destroy(target);
  OrbRuntime_destroy(rt.ok);
  printf("smoke.c: ok\n");
  return 0;
}
