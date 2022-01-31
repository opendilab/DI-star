import collections
from absl import logging
import random

import time
import enum
import numpy as np
import six
import torch
import copy
import random

from concurrent.futures import ThreadPoolExecutor
from distar.pysc2.lib import actions
from distar.pysc2.lib import point, colors, static_data
from distar.pysc2.lib import named_array
from distar.pysc2.lib import stopwatch
from distar.pysc2.lib.static_data import BUFFS_REORDER, BUFFS_REORDER_ARRAY, UPGRADES_REORDER_ARRAY, ADDON_REORDER_ARRAY, \
    UNIT_TYPES_REORDER_ARRAY, NUM_UNIT_TYPES, NUM_UPGRADES

from s2clientprotocol import raw_pb2 as sc_raw
from distar.pysc2.lib import unit_controls
from s2clientprotocol import sc2api_pb2 as sc_pb
from torch import int8, uint8, int16, int32, float32, float16, int64

from .actions import ACTIONS, ABILITY_TO_QUEUE_ACTION, BEGINNING_ORDER_ACTIONS, CUMULATIVE_STAT_ACTIONS, FUNC_ID_TO_ACTION_TYPE_DICT, UNIT_ABILITY_REORDER, NUM_UNIT_MIX_ABILITIES
from collections import defaultdict

sw = stopwatch.sw

SPATIAL_SIZE = [152, 160]  # y, x
BUFF_LENGTH = 3
UPGRADE_LENGTH = 20
MAX_DELAY = 127
BEGINNING_ORDER_LENGTH = 20
MAX_SELECTED_UNITS_NUM = 64
MAX_ENTITY_NUM = 512
EFFECT_LEN = 100


SPATIAL_INFO = [('height_map', uint8), ('visibility_map', uint8), ('creep', uint8), ('player_relative', uint8),
                ('alerts', uint8), ('pathable', uint8), ('buildable', uint8), ('effect_PsiStorm', int16),
                ('effect_NukeDot', int16), ('effect_LiberatorDefenderZone', int16), ('effect_BlindingCloud', int16),
                ('effect_CorrosiveBile', int16), ('effect_LurkerSpines', int16)]

# (name, dtype, size)
SCALAR_INFO = [('home_race', uint8, ()), ('away_race', uint8, ()), ('upgrades', int16, (NUM_UPGRADES,)),
               ('time', float32, ()), ('unit_counts_bow', uint8, (NUM_UNIT_TYPES,)),
               ('agent_statistics', float32, (10, )),
               ('cumulative_stat', uint8, (len(CUMULATIVE_STAT_ACTIONS), )),
               ('beginning_order', int16, (BEGINNING_ORDER_LENGTH, )), ('last_queued', int16, ()),
               ('last_delay', int16, ()), ('last_action_type', int16, ()),
               ('bo_location', int16, (BEGINNING_ORDER_LENGTH, )),
               ('unit_order_type', uint8, (NUM_UNIT_MIX_ABILITIES,)), ('unit_type_bool', uint8, (NUM_UNIT_TYPES,)),
               ('enemy_unit_type_bool', uint8, (NUM_UNIT_TYPES,))]

ENTITY_INFO = [('unit_type', int16), ('alliance', uint8), ('cargo_space_taken', uint8),
               ('build_progress', float16), ('health_ratio', float16), ('shield_ratio', float16),
               ('energy_ratio', float16), ('display_type', uint8), ('x', uint8), ('y', uint8),
               ('cloak', uint8), ('is_blip', uint8), ('is_powered', uint8), ('mineral_contents', float16),
               ('vespene_contents', float16), ('cargo_space_max', uint8), ('assigned_harvesters', uint8),
               ('weapon_cooldown', uint8), ('order_length', uint8), ('order_id_0', int16),
               ('order_id_1', int16), ('is_hallucination', uint8), ('buff_id_0', uint8), ('buff_id_1', uint8),
               ('addon_unit_type', uint8), ('is_active', uint8), ('order_progress_0', float16),
               ('order_progress_1', float16), ('order_id_2', int16), ('order_id_3', int16),
               ('is_in_cargo', uint8), ('attack_upgrade_level', uint8), ('armor_upgrade_level', uint8),
               ('shield_upgrade_level', uint8), ('last_selected_units', int8), ('last_targeted_unit', int8)]

ACTION_INFO = {'action_type': torch.tensor(0, dtype=torch.long), 'delay': torch.tensor(0, dtype=torch.long),
               'queued': torch.tensor(0, dtype=torch.long), 'selected_units': torch.zeros((MAX_SELECTED_UNITS_NUM,), dtype=torch.long),
               'target_unit': torch.tensor(0, dtype=torch.long),
               'target_location': torch.tensor(0, dtype=torch.long)}

ACTION_LOGP = {'action_type': torch.tensor(0, dtype=torch.float), 'delay': torch.tensor(0, dtype=torch.float),
               'queued': torch.tensor(0, dtype=torch.float), 'selected_units': torch.zeros((MAX_SELECTED_UNITS_NUM,), dtype=torch.float),
               'target_unit': torch.tensor(0, dtype=torch.float),
               'target_location': torch.tensor(0, dtype=torch.float)}

ACTION_LOGIT = {'action_type': torch.zeros(len(ACTIONS), dtype=torch.float), 'delay': torch.zeros(MAX_DELAY + 1, dtype=torch.float),
                'queued': torch.zeros(2, dtype=torch.float), 'selected_units': torch.zeros((MAX_SELECTED_UNITS_NUM, MAX_ENTITY_NUM + 1), dtype=torch.float),
                'target_unit': torch.zeros(MAX_ENTITY_NUM, dtype=torch.float),
                'target_location': torch.zeros(SPATIAL_SIZE[0] * SPATIAL_SIZE[1], dtype=torch.float)}


def recursive_to_share_memory(data, batch_size):
    if isinstance(data, torch.Tensor):
        if batch_size is not None:
            data_shape_len = len(data.shape)
            data = data.repeat(batch_size, *([1] * data_shape_len))
        return data.share_memory_()
    elif isinstance(data, dict):
        return {k: recursive_to_share_memory(v, batch_size) for k, v in data.items()}


def fake_step_data(share_memory=False, batch_size=None, train=True, hidden_size=None, hidden_layer=None):
    spatial_info = {}
    scalar_info = {}
    entity_info = {}
    for k, dtype in SPATIAL_INFO:
        if 'effect' in k:
            spatial_info[k] = torch.zeros(EFFECT_LEN, dtype=dtype)
        else:
            spatial_info[k] = torch.zeros(size=SPATIAL_SIZE, dtype=dtype)
    for k, dtype, size in SCALAR_INFO:
        scalar_info[k] = torch.zeros(size=size, dtype=dtype)
    for k, dtype in ENTITY_INFO:
        entity_info[k] = torch.zeros(size=(MAX_ENTITY_NUM, ), dtype=dtype)
    action_mask = {'action_type': torch.tensor(1, dtype=torch.bool), 'delay': torch.tensor(1, dtype=torch.bool),
                   'queued': torch.tensor(1, dtype=torch.bool), 'selected_units': torch.tensor(1, dtype=torch.bool),
                   'target_unit': torch.tensor(1, dtype=torch.bool),
                   'target_location': torch.tensor(1, dtype=torch.bool)}
    ret = {
        'spatial_info': spatial_info,
        'scalar_info': scalar_info,
        'entity_info': entity_info,
        'entity_num': torch.randint(0, MAX_ENTITY_NUM, size=(), dtype=torch.long),
    }
    if train:
        ret.update({'action_info': copy.deepcopy(ACTION_INFO),
                    'action_mask': action_mask,
                    'selected_units_num': torch.randint(0, MAX_SELECTED_UNITS_NUM, size=(), dtype=torch.long)})
    if share_memory:
        ret = recursive_to_share_memory(ret, batch_size)
    if hidden_size is not None:
        ret['hidden_state'] = [(torch.zeros(batch_size, hidden_size).share_memory_(),
                                torch.zeros(batch_size, hidden_size).share_memory_()) for _ in range(hidden_layer)]
    return ret


def fake_model_output(batch_size, hidden_size, hidden_layer, teacher=False):
    ret = {
        'logit': copy.deepcopy(ACTION_LOGIT),
        'entity_num': torch.randint(0, MAX_ENTITY_NUM, size=(), dtype=torch.long),
        'selected_units_num': torch.randint(0, MAX_SELECTED_UNITS_NUM, size=(), dtype=torch.long)
    }
    if not teacher:
        ret.update({
            'action_info': copy.deepcopy(ACTION_INFO),
            'action_logp': copy.deepcopy(ACTION_LOGP),
            'extra_units': torch.zeros(MAX_ENTITY_NUM + 1),
        })
    ret = recursive_to_share_memory(ret, batch_size)
    ret['hidden_state'] = [(torch.zeros(batch_size, hidden_size).share_memory_(),
                            torch.zeros(batch_size, hidden_size).share_memory_()) for _ in range(hidden_layer)]
    return ret


class FeatureType(enum.Enum):
    SCALAR = 1
    CATEGORICAL = 2


class PlayerRelative(enum.IntEnum):
    """The values for the `player_relative` feature layers."""
    NONE = 0
    SELF = 1
    ALLY = 2
    NEUTRAL = 3
    ENEMY = 4


class Effects(enum.IntEnum):
    """Values for the `effects` feature layer."""
    # pylint: disable=invalid-name
    none = 0
    PsiStorm = 1
    GuardianShield = 2
    TemporalFieldGrowing = 3
    TemporalField = 4
    ThermalLance = 5
    ScannerSweep = 6
    NukeDot = 7
    LiberatorDefenderZoneSetup = 8
    LiberatorDefenderZone = 9
    BlindingCloud = 10
    CorrosiveBile = 11
    LurkerSpines = 12
    # pylint: enable=invalid-name


class ScoreCategories(enum.IntEnum):
  """Indices for the `score_by_category` observation's second dimension."""
  none = 0
  army = 1
  economy = 2
  technology = 3
  upgrade = 4


class Player(enum.IntEnum):
    """Indices into the `player` observation."""
    player_id = 0
    minerals = 1
    vespene = 2
    food_used = 3
    food_cap = 4
    food_army = 5
    food_workers = 6
    idle_worker_count = 7
    army_count = 8
    warp_gate_count = 9
    larva_count = 10


class FeatureUnit(enum.IntEnum):
    """Indices for the `feature_unit` observations."""
    unit_type = 0
    alliance = 1
    cargo_space_taken = 2
    build_progress = 3
    health_max = 4
    shield_max = 5
    energy_max = 6
    display_type = 7
    owner = 8
    x = 9
    y = 10
    cloak = 11
    is_blip = 12
    is_powered = 13
    mineral_contents = 14
    vespene_contents = 15
    cargo_space_max = 16
    assigned_harvesters = 17
    weapon_cooldown = 18
    order_length = 19  # If zero, the unit is idle.
    order_id_0 = 20
    order_id_1 = 21
    # tag = 22  # Unique identifier for a unit (only populated for raw units).
    is_hallucination = 22
    buff_id_0 = 23
    buff_id_1 = 24
    addon_unit_type = 25
    is_active = 26
    order_progress_0 = 27
    order_progress_1 = 28
    order_id_2 = 29
    order_id_3 = 30
    is_in_cargo = 31
    attack_upgrade_level = 32
    armor_upgrade_level = 33
    shield_upgrade_level = 34
    health = 35
    shield = 36
    energy = 37


class EffectPos(enum.IntEnum):
    """Positions of the active effects."""
    effect = 0
    alliance = 1
    owner = 2
    radius = 3
    x = 4
    y = 5


class Feature(collections.namedtuple(
    "Feature", ["index", "name", "layer_set", "full_name", "scale", "type",
                "palette", "clip"])):
    """Define properties of a feature layer.

    Attributes:
      index: Index of this layer into the set of layers.
      name: The name of the layer within the set.
      layer_set: Which set of feature layers to look at in the observation proto.
      full_name: The full name including for visualization.
      scale: Max value (+1) of this layer, used to scale the values.
      type: A FeatureType for scalar vs categorical.
      palette: A color palette for rendering.
      clip: Whether to clip the values for coloring.
    """
    __slots__ = ()

    dtypes = {
        1: np.uint8,
        8: np.uint8,
        16: np.uint16,
        32: np.int32,
    }

    def unpack(self, obs):
        """Return a correctly shaped numpy array for this feature."""
        planes = getattr(obs.feature_layer_data, self.layer_set)
        plane = getattr(planes, self.name)
        return self.unpack_layer(plane)

    @staticmethod
    @sw.decorate
    def unpack_layer(plane):
        """Return a correctly shaped numpy array given the feature layer bytes."""
        size = point.Point.build(plane.size)
        if size == (0, 0):
            # New layer that isn't implemented in this SC2 version.
            return None
        data = np.frombuffer(plane.data, dtype=Feature.dtypes[plane.bits_per_pixel])
        if plane.bits_per_pixel == 1:
            data = np.unpackbits(data)
            if data.shape[0] != size.x * size.y:
                # This could happen if the correct length isn't a multiple of 8, leading
                # to some padding bits at the end of the string which are incorrectly
                # interpreted as data.
                data = data[:size.x * size.y]
        return data.reshape(size.y, size.x)

    @staticmethod
    @sw.decorate
    def unpack_rgb_image(plane):
        """Return a correctly shaped numpy array given the image bytes."""
        assert plane.bits_per_pixel == 24, "{} != 24".format(plane.bits_per_pixel)
        size = point.Point.build(plane.size)
        data = np.frombuffer(plane.data, dtype=np.uint8)
        return data.reshape(size.y, size.x, 3)

    @sw.decorate
    def color(self, plane):
        if self.clip:
            plane = np.clip(plane, 0, self.scale - 1)
        return self.palette[plane]


class MinimapFeatures(collections.namedtuple("MinimapFeatures", [
    "height_map", "visibility_map", "creep", "player_relative", "alerts", "pathable", "buildable"])):
    """The set of minimap feature layers."""
    __slots__ = ()

    def __new__(cls, **kwargs):
        feats = {}
        for name, (scale, type_, palette) in six.iteritems(kwargs):
            feats[name] = Feature(
                index=MinimapFeatures._fields.index(name),
                name=name,
                layer_set="minimap_renders",
                full_name="minimap " + name,
                scale=scale,
                type=type_,
                palette=palette(scale) if callable(palette) else palette,
                clip=False)
        return super(MinimapFeatures, cls).__new__(cls, **feats)  # pytype: disable=missing-parameter


MINIMAP_FEATURES = MinimapFeatures(
    height_map=(256, FeatureType.SCALAR, colors.height_map),
    visibility_map=(4, FeatureType.CATEGORICAL, colors.VISIBILITY_PALETTE),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE),
    player_relative=(5, FeatureType.CATEGORICAL,
                     colors.PLAYER_RELATIVE_PALETTE),
    alerts=(2, FeatureType.CATEGORICAL, colors.winter),
    pathable=(2, FeatureType.CATEGORICAL, colors.winter),
    buildable=(2, FeatureType.CATEGORICAL, colors.winter),
)


def compute_battle_score(obs):
    if obs is None:
        return 0.
    score_details = obs.observation.score.score_details
    killed_mineral, killed_vespene = 0., 0.
    for s in ScoreCategories:
        killed_mineral += getattr(score_details.killed_minerals, s.name)
        killed_vespene += getattr(score_details.killed_vespene, s.name)
    battle_score = killed_mineral + 1.5 * killed_vespene
    return battle_score


class Features(object):
    def __init__(self, game_info, raw_ob, cfg={}):
        self._map_size = game_info.start_raw.map_size
        self._requested_races = {
            info.player_id: info.race_requested for info in game_info.player_info
            if info.type != sc_pb.Observer}
        self._map_name = game_info.map_name
        self._start_location = game_info.start_raw.start_locations[0]

        self._whole_cfg = cfg
        self._cfg = cfg.get('feature', {})
        self._bo_zergling_num = self._cfg.get('bo_zergling_num', 8)
        self._beginning_order_flag = random.random() < self._cfg.get('beginning_order_prob', 1.)
        self._cumulative_stat_flag = random.random() < self._cfg.get('cumulative_stat_prob', 1.)
        self._zero_z_value = self._cfg.get('zero_z_value', 1.)
        self._filter_spine = self._cfg.get('filter_spine', True)
        self._init_born_location(game_info, raw_ob)

    def _init_born_location(self, game_info, raw_ob):
        location = []
        for i in raw_ob.observation.raw_data.units:
            if i.unit_type == 59 or i.unit_type == 18 or i.unit_type == 86:
                location.append([i.pos.x, i.pos.y])
        assert len(location) == 1, 'this replay is corrupt, no fog of war, check replays from this game version'
        born_location = location[0]
        self._born_location = int(born_location[0]) + int(self.map_size.y - born_location[1]) * SPATIAL_SIZE[1]
        away_born_location = game_info.start_raw.start_locations[0]
        self._away_born_location = int(away_born_location.x) + int(self.map_size.y - away_born_location.y) * SPATIAL_SIZE[1]

    @property
    def home_born_location(self):
        return self._born_location

    @property
    def away_born_location(self):
        return self._away_born_location

    @property
    def start_location(self):
        return self._start_location

    @property
    def map_name(self):
        return self._map_name

    @property
    def map_size(self):
        return self._map_size

    @property
    def requested_races(self):
        return self._requested_races

    def get_z(self, traj_data):
        zergling_count = 0
        beginning_order = []
        bo_location = []
        cumulative_stat = torch.zeros(len(CUMULATIVE_STAT_ACTIONS), dtype=torch.int8)
        own_x = self.home_born_location % SPATIAL_SIZE[1]
        own_y = self.home_born_location // SPATIAL_SIZE[1]
        away_x = self.away_born_location % SPATIAL_SIZE[1]
        away_y = self.away_born_location // SPATIAL_SIZE[1]
        strategy_flag = 0.
        for step_data in traj_data:
            action_type = step_data['action_info']['action_type'].item()
            if action_type == 322:
                zergling_count += 1
                if zergling_count > self._bo_zergling_num:
                    continue
            if action_type in BEGINNING_ORDER_ACTIONS:
                location = step_data['action_info']['target_location'].item()
                if self._filter_spine and action_type == 54:
                    x, y = location % SPATIAL_SIZE[1], location // SPATIAL_SIZE[1]
                    own_distance = (own_x - x) ** 2 + (own_y - y) ** 2
                    away_distance = (away_x - x) ** 2 + (away_y - y) ** 2
                    if own_distance < away_distance:
                        continue
                beginning_order.append(BEGINNING_ORDER_ACTIONS.index(action_type))
                bo_location.append(location)
            if action_type in CUMULATIVE_STAT_ACTIONS:
                cumulative_stat[CUMULATIVE_STAT_ACTIONS.index(action_type)] = 1
        bo_len = len(beginning_order)
        if bo_len < BEGINNING_ORDER_LENGTH:
            beginning_order += [0] * (BEGINNING_ORDER_LENGTH - bo_len)
            bo_location += [0] * (BEGINNING_ORDER_LENGTH - bo_len)
        else:
            beginning_order = beginning_order[:BEGINNING_ORDER_LENGTH]
            bo_location = bo_location[:BEGINNING_ORDER_LENGTH]
        beginning_order = torch.as_tensor(beginning_order, dtype=torch.short)
        bo_location = torch.as_tensor(bo_location, dtype=torch.short)
        beginning_order = self._beginning_order_flag * beginning_order
        bo_location = self._beginning_order_flag * bo_location
        if not self._cumulative_stat_flag:
            cumulative_stat = 0 * cumulative_stat + self._zero_z_value
        return beginning_order, cumulative_stat, bo_len, bo_location

    @sw.decorate
    def transform_obs(self, obs, padding_spatial=False, opponent_obs=None):
        spatial_info = defaultdict(list)
        scalar_info = {}
        entity_info = dict()
        game_info = {}

        raw = obs.observation.raw_data
        # spatial info
        for f in MINIMAP_FEATURES:
            d = f.unpack(obs.observation).copy()
            d = torch.from_numpy(d)
            padding_y = SPATIAL_SIZE[0] - d.shape[0]
            padding_x = SPATIAL_SIZE[1] - d.shape[1]
            if (padding_y != 0 or padding_x != 0) and padding_spatial:
                d = torch.nn.functional.pad(d, (0, padding_x, 0, padding_y), 'constant', 0)
            spatial_info[f.name] = d
        for e in raw.effects:
            name = Effects(e.effect_id).name
            if name in ['LiberatorDefenderZone', 'LurkerSpines'] and e.owner == 1:
                continue
            for p in e.pos:
                location = int(p.x) + int(self.map_size.y - p.y) * SPATIAL_SIZE[1]
                spatial_info['effect_' + name].append(location)
        for k, _ in SPATIAL_INFO:
            if 'effect' in k:
                padding_num = EFFECT_LEN - len(spatial_info[k])
                if padding_num > 0:
                    spatial_info[k] += [0] * padding_num
                else:
                    spatial_info[k] = spatial_info[k][:EFFECT_LEN]
                spatial_info[k] = torch.as_tensor(spatial_info[k], dtype=int16)

        # entity info
        tag_types = {}  # Only populate the cache if it's needed.
        def get_addon_type(tag):
            if not tag_types:
                for u in raw.units:
                    tag_types[u.tag] = u.unit_type
            return tag_types.get(tag, 0)
        tags = []
        units = []
        for u in raw.units:
            tags.append(u.tag)
            units.append([
                u.unit_type,
                u.alliance,  # Self = 1, Ally = 2, Neutral = 3, Enemy = 4
                u.cargo_space_taken,
                u.build_progress,
                u.health_max,
                u.shield_max,
                u.energy_max,
                u.display_type,  # Visible = 1, Snapshot = 2, Hidden = 3
                u.owner,  # 1-15, 16 = neutral
                u.pos.x,
                u.pos.y,
                u.cloak,  # Cloaked = 1, CloakedDetected = 2, NotCloaked = 3
                u.is_blip,
                u.is_powered,
                u.mineral_contents,
                u.vespene_contents,
                # Not populated for enemies or neutral
                u.cargo_space_max,
                u.assigned_harvesters,
                u.weapon_cooldown,
                len(u.orders),
                u.orders[0].ability_id if len(u.orders) > 0 else 0,
                u.orders[1].ability_id if len(u.orders) > 1 else 0,
                u.is_hallucination,
                u.buff_ids[0] if len(u.buff_ids) >= 1 else 0,
                u.buff_ids[1] if len(u.buff_ids) >= 2 else 0,
                get_addon_type(u.add_on_tag) if u.add_on_tag else 0,
                u.is_active,
                u.orders[0].progress if len(u.orders) >= 1 else 0,
                u.orders[1].progress if len(u.orders) >= 2 else 0,
                u.orders[2].ability_id if len(u.orders) > 2 else 0,
                u.orders[3].ability_id if len(u.orders) > 3 else 0,
                0,
                u.attack_upgrade_level,
                u.armor_upgrade_level,
                u.shield_upgrade_level,
                u.health,
                u.shield,
                u.energy,
            ])
            for v in u.passengers:
                tags.append(v.tag)
                units.append([
                    v.unit_type,
                    u.alliance,  # Self = 1, Ally = 2, Neutral = 3, Enemy = 4
                    0,
                    0,
                    v.health_max,
                    v.shield_max,
                    v.energy_max,
                    0,  # Visible = 1, Snapshot = 2, Hidden = 3
                    u.owner,  # 1-15, 16 = neutral
                    u.pos.x,
                    u.pos.y,
                    0,  # Cloaked = 1, CloakedDetected = 2, NotCloaked = 3
                    0,
                    0,
                    0,
                    0,
                    # Not populated for enemies or neutral
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    v.health,
                    v.shield,
                    v.energy,
                ])
        units = units[:MAX_ENTITY_NUM]
        tags = tags[:MAX_ENTITY_NUM]
        raw_entity_info = named_array.NamedNumpyArray(units, [None, FeatureUnit], dtype=np.float32)

        for k, dtype in ENTITY_INFO:
            if 'last' in k:
                pass
            elif k == 'unit_type':
                entity_info[k] = UNIT_TYPES_REORDER_ARRAY[raw_entity_info[:, 'unit_type']].short()
            elif 'order_id' in k:
                order_idx = int(k.split('_')[-1])
                if order_idx == 0:
                    entity_info[k] = UNIT_ABILITY_REORDER[raw_entity_info[:, k]].short()
                    invalid_actions = entity_info[k] == -1
                    if invalid_actions.any():
                       print('[ERROR] invalid unit ability', raw_entity_info[invalid_actions, k])
                else:
                    entity_info[k] = ABILITY_TO_QUEUE_ACTION[raw_entity_info[:, k]].short()
                    invalid_actions = entity_info[k] == -1
                    if invalid_actions.any():
                       print('[ERROR] invalid queue ability', raw_entity_info[invalid_actions, k])
            elif 'buff_id' in k:
                entity_info[k] = BUFFS_REORDER_ARRAY[raw_entity_info[:, k]].short()
            elif k == 'addon_unit_type':
                entity_info[k] = ADDON_REORDER_ARRAY[raw_entity_info[:, k]].short()
            elif k == 'cargo_space_taken':
                entity_info[k] = torch.as_tensor(raw_entity_info[:, 'cargo_space_taken'], dtype=dtype).clamp_(min=0, max=8)
            elif k == 'cargo_space_max':
                entity_info[k] = torch.as_tensor(raw_entity_info[:, 'cargo_space_max'], dtype=dtype).clamp_(min=0, max=8)
            elif k == 'health_ratio':
                entity_info[k] = torch.as_tensor(raw_entity_info[:, 'health'], dtype=dtype) / (torch.as_tensor(raw_entity_info[:, 'health_max'], dtype=dtype) + 1e-6)
            elif k == 'shield_ratio':
                entity_info[k] = torch.as_tensor(raw_entity_info[:, 'shield'], dtype=dtype) / (torch.as_tensor(raw_entity_info[:, 'shield_max'], dtype=dtype) + 1e-6)
            elif k == 'energy_ratio':
                entity_info[k] = torch.as_tensor(raw_entity_info[:, 'energy'], dtype=dtype) / (torch.as_tensor(raw_entity_info[:, 'energy_max'], dtype=dtype) + 1e-6)
            elif k == 'mineral_contents':
                entity_info[k] = torch.as_tensor(raw_entity_info[:, 'mineral_contents'], dtype=dtype) / 1800
            elif k == 'vespene_contents':
                entity_info[k] = torch.as_tensor(raw_entity_info[:, 'vespene_contents'], dtype=dtype) / 2500
            elif k == 'y':
                entity_info[k] = torch.as_tensor(self.map_size.y -  raw_entity_info[:, 'y'], dtype=dtype)
            else:
                entity_info[k] = torch.as_tensor(raw_entity_info[:, k], dtype=dtype)

        # scalar info
        scalar_info['time'] = torch.tensor(obs.observation.game_loop, dtype=torch.float)
        player = obs.observation.player_common
        scalar_info['agent_statistics'] = torch.tensor([
            player.minerals,
            player.vespene,
            player.food_used,
            player.food_cap,
            player.food_army,
            player.food_workers,
            player.idle_worker_count,
            player.army_count,
            player.warp_gate_count,
            player.larva_count], dtype=torch.float)
        scalar_info['agent_statistics'] = torch.log(scalar_info['agent_statistics'] + 1)

        scalar_info["home_race"] = torch.tensor(
            self._requested_races[player.player_id], dtype=torch.uint8)
        for player_id, race in self._requested_races.items():
            if player_id != player.player_id:
                scalar_info["away_race"] = torch.tensor(race, dtype=torch.uint8)

        upgrades = torch.zeros(NUM_UPGRADES, dtype=torch.uint8)
        raw_upgrades = UPGRADES_REORDER_ARRAY[raw.player.upgrade_ids[:UPGRADE_LENGTH]]
        # for u in raw.player.upgrade_ids:
            #if UPGRADES_REORDER_ARRAY[u] == -1:
            #    print('[ERROR]', u)
        upgrades.scatter_(dim=0, index=raw_upgrades, value=1.)
        scalar_info["upgrades"] = upgrades

        unit_counts_bow = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
        scalar_info['unit_type_bool'] = torch.zeros(NUM_UNIT_TYPES, dtype=uint8)
        own_unit_types = entity_info['unit_type'][entity_info['alliance'] == 1]
        scalar_info['unit_counts_bow'] = torch.scatter_add(unit_counts_bow, dim=0, index=own_unit_types.long(), src=torch.ones_like(own_unit_types, dtype=torch.uint8))
        scalar_info['unit_type_bool'] = (scalar_info['unit_counts_bow'] > 0).to(uint8)

        scalar_info['unit_order_type'] = torch.zeros(NUM_UNIT_MIX_ABILITIES, dtype=uint8)
        own_unit_orders = entity_info['order_id_0'][entity_info['alliance'] == 1]
        scalar_info['unit_order_type'].scatter_(0, own_unit_orders.long(), torch.ones_like(own_unit_orders, dtype=uint8))

        enemy_unit_types = entity_info['unit_type'][entity_info['alliance'] == 4]
        enemy_unit_type_bool = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
        scalar_info['enemy_unit_type_bool'] = torch.scatter(enemy_unit_type_bool, dim=0, index=enemy_unit_types.long(), src=torch.ones_like(enemy_unit_types, dtype=torch.uint8))

        # game info
        game_info['map_name'] = self._map_name
        game_info['action_result'] = [o.result for o in obs.action_errors]
        game_info['game_loop'] = obs.observation.game_loop
        game_info['tags'] = tags
        game_info['battle_score'] = compute_battle_score(obs)
        game_info['opponent_battle_score'] = 0.
        ret = {
            'spatial_info': spatial_info, 'scalar_info': scalar_info, 'entity_num': torch.tensor(len(entity_info['unit_type']), dtype=torch.long),
            'entity_info': entity_info, 'game_info': game_info,
            }

        # value feature
        if opponent_obs:
            raw = opponent_obs.observation.raw_data
            enemy_unit_counts_bow = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
            enemy_x = []
            enemy_y = []
            enemy_unit_type = []
            unit_alliance = []
            for u in raw.units:
                if u.alliance == 1:
                    enemy_x.append(u.pos.x)
                    enemy_y.append(u.pos.y)
                    enemy_unit_type.append(u.unit_type)
                    unit_alliance.append(1)
            enemy_unit_type = UNIT_TYPES_REORDER_ARRAY[enemy_unit_type].short()
            enemy_unit_counts_bow = torch.scatter_add(enemy_unit_counts_bow, dim=0, index=enemy_unit_type.long(),
                                                      src=torch.ones_like(enemy_unit_type, dtype=torch.uint8))
            enemy_unit_type_bool = (enemy_unit_counts_bow > 0).to(uint8)


            unit_type = torch.cat([enemy_unit_type, own_unit_types], dim=0)
            enemy_x = torch.as_tensor(enemy_x, dtype=uint8)
            unit_x = torch.cat([enemy_x, entity_info['x'][entity_info['alliance'] == 1]], dim=0)

            enemy_y = torch.as_tensor(enemy_y, dtype=float32)
            enemy_y = torch.as_tensor(self.map_size.y - enemy_y, dtype=uint8)
            unit_y = torch.cat([enemy_y, entity_info['y'][entity_info['alliance'] == 1]], dim=0)
            total_unit_count = len(unit_y)
            unit_alliance += [0] * (total_unit_count - len(unit_alliance))
            unit_alliance = torch.as_tensor(unit_alliance, dtype=torch.bool)

            padding_num = MAX_ENTITY_NUM - total_unit_count
            if padding_num > 0:
                unit_x = torch.nn.functional.pad(unit_x, (0, padding_num), 'constant', 0)
                unit_y = torch.nn.functional.pad(unit_y, (0, padding_num), 'constant', 0)
                unit_type = torch.nn.functional.pad(unit_type, (0, padding_num), 'constant', 0)
                unit_alliance = torch.nn.functional.pad(unit_alliance, (0, padding_num), 'constant', 0)
            else:
                unit_x = unit_x[:MAX_ENTITY_NUM]
                unit_y = unit_y[:MAX_ENTITY_NUM]
                unit_type = unit_type[:MAX_ENTITY_NUM]
                unit_alliance = unit_alliance[:MAX_ENTITY_NUM]
            
            total_unit_count = torch.tensor(total_unit_count, dtype=torch.long)

            player = opponent_obs.observation.player_common
            enemy_agent_statistics = torch.tensor([
                player.minerals,
                player.vespene,
                player.food_used,
                player.food_cap,
                player.food_army,
                player.food_workers,
                player.idle_worker_count,
                player.army_count,
                player.warp_gate_count,
                player.larva_count], dtype=torch.float)
            enemy_agent_statistics = torch.log(enemy_agent_statistics + 1)
            enemy_raw_upgrades = UPGRADES_REORDER_ARRAY[raw.player.upgrade_ids[:UPGRADE_LENGTH]]
            enemy_upgrades = torch.zeros(NUM_UPGRADES, dtype=torch.uint8)
            enemy_upgrades.scatter_(dim=0, index=enemy_raw_upgrades, value=1.)

            d = MINIMAP_FEATURES.player_relative.unpack(opponent_obs.observation).copy()
            d = torch.from_numpy(d)
            padding_y = SPATIAL_SIZE[0] - d.shape[0]
            padding_x = SPATIAL_SIZE[1] - d.shape[1]
            if (padding_y != 0 or padding_x != 0) and padding_spatial:
                d = torch.nn.functional.pad(d, (0, padding_x, 0, padding_y), 'constant', 0)
            enemy_units_spatial = d == 1
            own_units_spatial = ret['spatial_info']['player_relative'] == 1
            value_feature = {'unit_type': unit_type, 'enemy_unit_counts_bow': enemy_unit_counts_bow,
                             'enemy_unit_type_bool': enemy_unit_type_bool, 'unit_x': unit_x, 'unit_y': unit_y,
                             'unit_alliance': unit_alliance, 'total_unit_count': total_unit_count,
                             'enemy_agent_statistics': enemy_agent_statistics, 'enemy_upgrades': enemy_upgrades,
                             'own_units_spatial': own_units_spatial.unsqueeze(dim=0), 'enemy_units_spatial': enemy_units_spatial.unsqueeze(dim=0)}
            ret['value_feature'] = value_feature
            game_info['opponent_battle_score'] = compute_battle_score(opponent_obs)
        return ret

    @sw.decorate
    def transform_action(self, func_call):
        """Transform an agent-style action to one that SC2 can consume.

        Args:
          obs: a `sc_pb.Observation` from the previous frame.
          func_call: a `FunctionCall` to be turned into a `sc_pb.Action`.
          skip_available: If True, assume the action is available. This should only
              be used for testing or if you expect to make actions that weren't
              valid at the last observation.

        Returns:
          a corresponding `sc_pb.Action`.

        Raises:
          ValueError: if the action doesn't pass validation.
        """
        # Ignore sc_pb.Action's to make the env more flexible, eg raw actions.
        if isinstance(func_call, sc_pb.Action):
            return func_call

        func_id = func_call.function
        raw = True  # TODO(nyz) self._raw
        try:
            if raw:
                func = actions.RAW_FUNCTIONS[func_id]
            else:
                func = actions.FUNCTIONS[func_id]
        except KeyError:
            raise ValueError("Invalid function id: %s." % func_id)

        # Right number of args?
        if len(func_call.arguments) != len(func.args):
            raise ValueError(
                "Wrong number of arguments for function: %s, got: %s" % (
                    func, func_call.arguments))

        # Args are valid?
        aif = self._agent_interface_format
        for t, arg in zip(func.args, func_call.arguments):
            if t.count:
                if 1 <= len(arg) <= t.count:
                    continue
                else:
                    raise ValueError(
                        "Wrong number of values for argument of %s, got: %s" % (
                            func, func_call.arguments))

            if t.name in ("screen", "screen2"):
                sizes = aif.action_dimensions.screen
            elif t.name == "minimap":
                sizes = aif.action_dimensions.minimap
            elif t.name == "world":
                sizes = aif.raw_resolution
            else:
                sizes = t.sizes

            if len(sizes) != len(arg):
                raise ValueError(
                    "Wrong number of values for argument of %s, got: %s" % (
                        func, func_call.arguments))

            for s, a in zip(sizes, arg):
                if not np.all(0 <= a) and np.all(a < s):
                    raise ValueError("Argument is out of range for %s, got: %s" % (
                        func, func_call.arguments))

        # Convert them to python types.
        kwargs = {type_.name: type_.fn(a)
                  for type_, a in zip(func.args, func_call.arguments)}

        # Call the right callback to get an SC2 action proto.
        sc2_action = sc_pb.Action()
        kwargs["action"] = sc2_action
        if func.ability_id:
            kwargs["ability_id"] = func.ability_id

        if raw:
            actions.RAW_FUNCTIONS[func_id].function_type(**kwargs)
        else:
            kwargs["action_space"] = aif.action_space
            actions.FUNCTIONS[func_id].function_type(**kwargs)
        return sc2_action

    @sw.decorate
    def reverse_raw_action(self, action, raw_tags):
        action_ret = {'action_type': None, 'delay': torch.tensor(0, dtype=torch.long), 'queued': None, 'selected_units': None, 'target_unit': None, 'target_location': None}
        last_selected_unit_tags = None
        last_target_unit_tag = None
        invalid_action_flag = False
        units = []
        tags = []

        def transfer_action_type(ability_id, cmd_type):
            cancel_slot = {313, 1039, 305, 307, 309, 1832, 1834, 3672}
            unload_unit = {410, 415, 397, 1440, 2373, 1409, 914, 3670}
            frivolous = {6, 7}  # Dance and Cheer
            if ability_id in frivolous:
                return None
            elif ability_id in unload_unit:
                ability_id = 3664 # unload all
            elif ability_id in cancel_slot:
                ability_id = 3671  # cancel_slot to cancel_quick
            general_id = next(iter(actions.RAW_ABILITY_IDS[ability_id])).general_id
            if general_id:
                ability_id = general_id
            for func in actions.RAW_ABILITY_IDS[ability_id]:
                if func.function_type is cmd_type:
                    action_type = FUNC_ID_TO_ACTION_TYPE_DICT[func.id]
                    return action_type
            print('[ERROR] invalid action ability', ability_id)
            return None

        raw_act = action.action_raw
        if raw_act.HasField("unit_command"):
            uc = raw_act.unit_command
            ability_id = uc.ability_id
            queue_command = uc.queue_command
            action_ret['queued'] = torch.tensor(queue_command, dtype=torch.long)
            for t in uc.unit_tags:
                try:
                    unit_index = raw_tags.index(t)
                    units.append(unit_index)
                    tags.append(t)
                except ValueError:
                    pass

            if uc.HasField("target_unit_tag"):
                try:
                    action_ret['target_unit'] = torch.tensor(raw_tags.index(uc.target_unit_tag), dtype=torch.long)
                    last_target_unit_tag = uc.target_unit_tag
                except ValueError:
                    invalid_action_flag = True
                action_ret['action_type'] = transfer_action_type(ability_id, actions.raw_cmd_unit)
            elif uc.HasField("target_world_space_pos"):
                x = min(int(uc.target_world_space_pos.x), self.map_size.x - 1)
                y = min(self.map_size.y - int(uc.target_world_space_pos.y), self.map_size.y - 1)
                label = y * SPATIAL_SIZE[1] + x 
                action_ret['target_location'] = torch.tensor(label, dtype=torch.long)
                action_ret['action_type'] = transfer_action_type(ability_id, actions.raw_cmd_pt)
            else:
                action_ret['action_type'] = transfer_action_type(ability_id, actions.raw_cmd)

        if raw_act.HasField("toggle_autocast"):
            uc = raw_act.toggle_autocast
            ability_id = uc.ability_id
            action_ret['action_type'] = transfer_action_type(ability_id, actions.raw_autocast)
            for t in uc.unit_tags:
                try:
                    unit_index = raw_tags.index(t)
                    units.append(unit_index)
                    tags.append(t)
                except ValueError:
                    pass

        if action_ret['action_type'] is not None:
            action_ret['action_type'] = torch.tensor(action_ret['action_type'], dtype=torch.long)
        else:
            invalid_action_flag = True
        
        if len(units) and not invalid_action_flag:
            last_selected_unit_tags = tags
            units.append(len(raw_tags))  # add end flag
            action_ret['selected_units'] = torch.tensor(units, dtype=torch.long)
            selected_units_num = torch.tensor(len(action_ret['selected_units']), dtype=torch.long)
        else:
            invalid_action_flag = True
            selected_units_num = torch.tensor(0, dtype=torch.long)
            

        action_mask = {}
        for k, v in action_ret.items():
            if v is None:
                action_mask[k] = torch.tensor(0, dtype=torch.bool)
                if k == 'selected_units':
                    action_ret[k] = torch.tensor([0], dtype=torch.long)
                else:
                    action_ret[k] = ACTION_INFO[k]
            else:
                action_mask[k] = torch.tensor(1, dtype=torch.bool)
        action_ret['selected_units'] = action_ret['selected_units'][:MAX_SELECTED_UNITS_NUM]
        selected_units_num.clamp_(max=MAX_SELECTED_UNITS_NUM)

        return action_ret, action_mask, selected_units_num, last_selected_unit_tags, last_target_unit_tag, invalid_action_flag
