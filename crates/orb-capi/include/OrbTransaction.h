#ifndef OrbTransaction_H
#define OrbTransaction_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "OrbError.d.h"

#include "OrbTransaction.d.h"






OrbTransaction* OrbTransaction_create(void);

typedef struct OrbTransaction_from_commands_result {union {OrbTransaction* ok; OrbError err;}; bool is_ok;} OrbTransaction_from_commands_result;
OrbTransaction_from_commands_result OrbTransaction_from_commands(DiplomatStringView commands);

typedef struct OrbTransaction_set_paint_property_result {union { OrbError err;}; bool is_ok;} OrbTransaction_set_paint_property_result;
OrbTransaction_set_paint_property_result OrbTransaction_set_paint_property(OrbTransaction* self, DiplomatStringView layer, DiplomatStringView name, DiplomatStringView value);

typedef struct OrbTransaction_set_layout_property_result {union { OrbError err;}; bool is_ok;} OrbTransaction_set_layout_property_result;
OrbTransaction_set_layout_property_result OrbTransaction_set_layout_property(OrbTransaction* self, DiplomatStringView layer, DiplomatStringView name, DiplomatStringView value);

typedef struct OrbTransaction_set_filter_result {union { OrbError err;}; bool is_ok;} OrbTransaction_set_filter_result;
OrbTransaction_set_filter_result OrbTransaction_set_filter(OrbTransaction* self, DiplomatStringView layer, DiplomatStringView filter);

void OrbTransaction_set_layer_zoom_range(OrbTransaction* self, DiplomatStringView layer, double min, double max);

typedef struct OrbTransaction_add_layer_result {union { OrbError err;}; bool is_ok;} OrbTransaction_add_layer_result;
OrbTransaction_add_layer_result OrbTransaction_add_layer(OrbTransaction* self, DiplomatStringView layer, DiplomatStringView before);

void OrbTransaction_remove_layer(OrbTransaction* self, DiplomatStringView id);

void OrbTransaction_move_layer(OrbTransaction* self, DiplomatStringView id, DiplomatStringView before);

typedef struct OrbTransaction_add_source_result {union { OrbError err;}; bool is_ok;} OrbTransaction_add_source_result;
OrbTransaction_add_source_result OrbTransaction_add_source(OrbTransaction* self, DiplomatStringView id, DiplomatStringView source);

void OrbTransaction_remove_source(OrbTransaction* self, DiplomatStringView id);

typedef struct OrbTransaction_set_source_data_result {union { OrbError err;}; bool is_ok;} OrbTransaction_set_source_data_result;
OrbTransaction_set_source_data_result OrbTransaction_set_source_data(OrbTransaction* self, DiplomatStringView id, DiplomatStringView data);

typedef struct OrbTransaction_update_source_data_result {union { OrbError err;}; bool is_ok;} OrbTransaction_update_source_data_result;
OrbTransaction_update_source_data_result OrbTransaction_update_source_data(OrbTransaction* self, DiplomatStringView id, DiplomatStringView diff);

typedef struct OrbTransaction_set_global_state_property_result {union { OrbError err;}; bool is_ok;} OrbTransaction_set_global_state_property_result;
OrbTransaction_set_global_state_property_result OrbTransaction_set_global_state_property(OrbTransaction* self, DiplomatStringView name, DiplomatStringView value);

typedef struct OrbTransaction_set_feature_state_result {union { OrbError err;}; bool is_ok;} OrbTransaction_set_feature_state_result;
OrbTransaction_set_feature_state_result OrbTransaction_set_feature_state(OrbTransaction* self, DiplomatStringView source, DiplomatStringView source_layer, DiplomatStringView id, DiplomatStringView state);

void OrbTransaction_remove_feature_state(OrbTransaction* self, DiplomatStringView source, DiplomatStringView source_layer, DiplomatStringView id, DiplomatStringView key);

typedef struct OrbTransaction_set_style_result {union { OrbError err;}; bool is_ok;} OrbTransaction_set_style_result;
OrbTransaction_set_style_result OrbTransaction_set_style(OrbTransaction* self, DiplomatStringView style);

typedef struct OrbTransaction_set_style_property_result {union { OrbError err;}; bool is_ok;} OrbTransaction_set_style_property_result;
OrbTransaction_set_style_property_result OrbTransaction_set_style_property(OrbTransaction* self, DiplomatStringView property, DiplomatStringView value);

void OrbTransaction_destroy(OrbTransaction* self);





#endif // OrbTransaction_H
