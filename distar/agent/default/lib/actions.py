import numpy as np
import torch


ACTIONS = [
    {'func_id': 0, 'general_ability_id': None, 'goal': 'other', 'name': 'no_op', 'queued': False, 'selected_units': False, 'target_location': False, 'target_unit': False} ,
    {'func_id': 168, 'general_ability_id': None, 'goal': 'other', 'name': 'raw_move_camera', 'queued': False, 'selected_units': False, 'target_location': True, 'target_unit': False} ,
    {'func_id': 2, 'general_ability_id': 3674, 'goal': 'other', 'name': 'Attack_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 3, 'general_ability_id': 3674, 'goal': 'other', 'name': 'Attack_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 88, 'general_ability_id': 2082, 'goal': 'other', 'name': 'Behavior_BuildingAttackOff_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 87, 'general_ability_id': 2081, 'goal': 'other', 'name': 'Behavior_BuildingAttackOn_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 169, 'general_ability_id': 3677, 'goal': 'other', 'name': 'Behavior_CloakOff_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 172, 'general_ability_id': 3676, 'goal': 'other', 'name': 'Behavior_CloakOn_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 175, 'general_ability_id': 1693, 'goal': 'other', 'name': 'Behavior_GenerateCreepOff_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 176, 'general_ability_id': 1692, 'goal': 'other', 'name': 'Behavior_GenerateCreepOn_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 177, 'general_ability_id': 3689, 'goal': 'other', 'name': 'Behavior_HoldFireOff_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 180, 'general_ability_id': 3688, 'goal': 'other', 'name': 'Behavior_HoldFireOn_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 158, 'general_ability_id': 2376, 'goal': 'other', 'name': 'Behavior_PulsarBeamOff_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 159, 'general_ability_id': 2375, 'goal': 'other', 'name': 'Behavior_PulsarBeamOn_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 183, 'general_ability_id': 331, 'goal': 'build', 'name': 'Build_Armory_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 29} ,
    {'func_id': 36, 'general_ability_id': 882, 'goal': 'build', 'name': 'Build_Assimilator_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True, 'game_id': 61} ,
    {'func_id': 184, 'general_ability_id': 1162, 'goal': 'build', 'name': 'Build_BanelingNest_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 96} ,
    {'func_id': 185, 'general_ability_id': 321, 'goal': 'build', 'name': 'Build_Barracks_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 21} ,
    {'func_id': 186, 'general_ability_id': 324, 'goal': 'build', 'name': 'Build_Bunker_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 24} ,
    {'func_id': 187, 'general_ability_id': 318, 'goal': 'build', 'name': 'Build_CommandCenter_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 18} ,
    {'func_id': 188, 'general_ability_id': 3691, 'goal': 'build', 'name': 'Build_CreepTumor_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 87} ,
    {'func_id': 47, 'general_ability_id': 894, 'goal': 'build', 'name': 'Build_CyberneticsCore_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 72} ,
    {'func_id': 44, 'general_ability_id': 891, 'goal': 'build', 'name': 'Build_DarkShrine_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 69} ,
    {'func_id': 191, 'general_ability_id': 322, 'goal': 'build', 'name': 'Build_EngineeringBay_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 22} ,
    {'func_id': 192, 'general_ability_id': 1156, 'goal': 'build', 'name': 'Build_EvolutionChamber_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 90} ,
    {'func_id': 193, 'general_ability_id': 1154, 'goal': 'build', 'name': 'Build_Extractor_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True, 'game_id': 88} ,
    {'func_id': 194, 'general_ability_id': 328, 'goal': 'build', 'name': 'Build_Factory_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 27} ,
    {'func_id': 39, 'general_ability_id': 885, 'goal': 'build', 'name': 'Build_FleetBeacon_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 64} ,
    {'func_id': 38, 'general_ability_id': 884, 'goal': 'build', 'name': 'Build_Forge_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 63} ,
    {'func_id': 195, 'general_ability_id': 333, 'goal': 'build', 'name': 'Build_FusionCore_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 30} ,
    {'func_id': 37, 'general_ability_id': 883, 'goal': 'build', 'name': 'Build_Gateway_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 62} ,
    {'func_id': 196, 'general_ability_id': 327, 'goal': 'build', 'name': 'Build_GhostAcademy_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 26} ,
    {'func_id': 197, 'general_ability_id': 1152, 'goal': 'build', 'name': 'Build_Hatchery_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 86} ,
    {'func_id': 198, 'general_ability_id': 1157, 'goal': 'build', 'name': 'Build_HydraliskDen_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 91} ,
    {'func_id': 199, 'general_ability_id': 1160, 'goal': 'build', 'name': 'Build_InfestationPit_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 94} ,
    {'func_id': 200, 'general_ability_id': 1042, 'goal': 'build', 'name': 'Build_Interceptors_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 85} ,
    {'func_id': 66, 'general_ability_id': 1042, 'goal': 'build', 'name': 'Build_Interceptors_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 85} ,
    {'func_id': 201, 'general_ability_id': 1163, 'goal': 'build', 'name': 'Build_LurkerDen_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 504} ,
    {'func_id': 202, 'general_ability_id': 323, 'goal': 'build', 'name': 'Build_MissileTurret_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 23} ,
    {'func_id': 34, 'general_ability_id': 880, 'goal': 'build', 'name': 'Build_Nexus_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 59} ,
    {'func_id': 203, 'general_ability_id': 710, 'goal': 'build', 'name': 'Build_Nuke_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 58} ,
    {'func_id': 204, 'general_ability_id': 1161, 'goal': 'build', 'name': 'Build_NydusNetwork_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 95} ,
    {'func_id': 205, 'general_ability_id': 1768, 'goal': 'build', 'name': 'Build_NydusWorm_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 142} ,
    {'func_id': 41, 'general_ability_id': 887, 'goal': 'build', 'name': 'Build_PhotonCannon_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 66} ,
    {'func_id': 35, 'general_ability_id': 881, 'goal': 'build', 'name': 'Build_Pylon_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 60} ,
    {'func_id': 207, 'general_ability_id': 3683, 'goal': 'build', 'name': 'Build_Reactor_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 6} ,
    {'func_id': 206, 'general_ability_id': 3683, 'goal': 'build', 'name': 'Build_Reactor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 6} ,
    {'func_id': 214, 'general_ability_id': 320, 'goal': 'build', 'name': 'Build_Refinery_pt', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True, 'game_id': 20} ,
    {'func_id': 215, 'general_ability_id': 1165, 'goal': 'build', 'name': 'Build_RoachWarren_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 97} ,
    {'func_id': 45, 'general_ability_id': 892, 'goal': 'build', 'name': 'Build_RoboticsBay_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 70} ,
    {'func_id': 46, 'general_ability_id': 893, 'goal': 'build', 'name': 'Build_RoboticsFacility_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 71} ,
    {'func_id': 216, 'general_ability_id': 326, 'goal': 'build', 'name': 'Build_SensorTower_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 25} ,
    {'func_id': 48, 'general_ability_id': 895, 'goal': 'build', 'name': 'Build_ShieldBattery_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 1910} ,
    {'func_id': 217, 'general_ability_id': 1155, 'goal': 'build', 'name': 'Build_SpawningPool_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 89} ,
    {'func_id': 218, 'general_ability_id': 1166, 'goal': 'build', 'name': 'Build_SpineCrawler_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 98} ,
    {'func_id': 219, 'general_ability_id': 1158, 'goal': 'build', 'name': 'Build_Spire_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 92} ,
    {'func_id': 220, 'general_ability_id': 1167, 'goal': 'build', 'name': 'Build_SporeCrawler_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 99} ,
    {'func_id': 42, 'general_ability_id': 889, 'goal': 'build', 'name': 'Build_Stargate_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 67} ,
    {'func_id': 221, 'general_ability_id': 329, 'goal': 'build', 'name': 'Build_Starport_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 28} ,
    {'func_id': 95, 'general_ability_id': 2505, 'goal': 'build', 'name': 'Build_StasisTrap_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 732} ,
    {'func_id': 222, 'general_ability_id': 319, 'goal': 'build', 'name': 'Build_SupplyDepot_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 19} ,
    {'func_id': 224, 'general_ability_id': 3682, 'goal': 'build', 'name': 'Build_TechLab_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 5} ,
    {'func_id': 223, 'general_ability_id': 3682, 'goal': 'build', 'name': 'Build_TechLab_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 5} ,
    {'func_id': 43, 'general_ability_id': 890, 'goal': 'build', 'name': 'Build_TemplarArchive_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 68} ,
    {'func_id': 40, 'general_ability_id': 886, 'goal': 'build', 'name': 'Build_TwilightCouncil_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 65} ,
    {'func_id': 231, 'general_ability_id': 1159, 'goal': 'build', 'name': 'Build_UltraliskCavern_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 93} ,
    {'func_id': 232, 'general_ability_id': 3661, 'goal': 'effect', 'name': 'BurrowDown_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 247, 'general_ability_id': 3662, 'goal': 'other', 'name': 'BurrowUp_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 246, 'general_ability_id': 3662, 'goal': 'other', 'name': 'BurrowUp_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 98, 'general_ability_id': 3659, 'goal': 'other', 'name': 'Cancel_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 129, 'general_ability_id': 3671, 'goal': 'other', 'name': 'Cancel_Last_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 293, 'general_ability_id': 2067, 'goal': 'effect', 'name': 'Effect_Abduct_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 96, 'general_ability_id': 2544, 'goal': 'effect', 'name': 'Effect_AdeptPhaseShift_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 294, 'general_ability_id': 3753, 'goal': 'effect', 'name': 'Effect_AntiArmorMissile_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 295, 'general_ability_id': 1764, 'goal': 'effect', 'name': 'Effect_AutoTurret_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 296, 'general_ability_id': 2063, 'goal': 'effect', 'name': 'Effect_BlindingCloud_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 111, 'general_ability_id': 3687, 'goal': 'effect', 'name': 'Effect_Blink_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 112, 'general_ability_id': 3687, 'goal': 'effect', 'name': 'Effect_Blink_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 297, 'general_ability_id': 171, 'goal': 'effect', 'name': 'Effect_CalldownMULE_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 298, 'general_ability_id': 171, 'goal': 'effect', 'name': 'Effect_CalldownMULE_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 299, 'general_ability_id': 2324, 'goal': 'effect', 'name': 'Effect_CausticSpray_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 302, 'general_ability_id': 1819, 'goal': 'effect', 'name': 'Effect_Charge_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 300, 'general_ability_id': 1819, 'goal': 'effect', 'name': 'Effect_Charge_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 301, 'general_ability_id': 1819, 'goal': 'effect', 'name': 'Effect_Charge_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 122, 'general_ability_id': 3755, 'goal': 'effect', 'name': 'Effect_ChronoBoostEnergyCost_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 33, 'general_ability_id': 261, 'goal': 'effect', 'name': 'Effect_ChronoBoost_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 303, 'general_ability_id': 1825, 'goal': 'effect', 'name': 'Effect_Contaminate_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 304, 'general_ability_id': 2338, 'goal': 'effect', 'name': 'Effect_CorrosiveBile_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 305, 'general_ability_id': 1628, 'goal': 'effect', 'name': 'Effect_EMP_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 306, 'general_ability_id': 1628, 'goal': 'effect', 'name': 'Effect_EMP_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 307, 'general_ability_id': 42, 'goal': 'effect', 'name': 'Effect_Explode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 157, 'general_ability_id': 140, 'goal': 'effect', 'name': 'Effect_Feedback_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 79, 'general_ability_id': 1526, 'goal': 'effect', 'name': 'Effect_ForceField_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 308, 'general_ability_id': 74, 'goal': 'effect', 'name': 'Effect_FungalGrowth_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 309, 'general_ability_id': 74, 'goal': 'effect', 'name': 'Effect_FungalGrowth_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 310, 'general_ability_id': 2714, 'goal': 'effect', 'name': 'Effect_GhostSnipe_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 32, 'general_ability_id': 173, 'goal': 'effect', 'name': 'Effect_GravitonBeam_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 20, 'general_ability_id': 76, 'goal': 'effect', 'name': 'Effect_GuardianShield_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 312, 'general_ability_id': 386, 'goal': 'effect', 'name': 'Effect_Heal_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 311, 'general_ability_id': 386, 'goal': 'effect', 'name': 'Effect_Heal_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 313, 'general_ability_id': 2328, 'goal': 'effect', 'name': 'Effect_ImmortalBarrier_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 91, 'general_ability_id': 2328, 'goal': 'effect', 'name': 'Effect_ImmortalBarrier_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 314, 'general_ability_id': 247, 'goal': 'effect', 'name': 'Effect_InfestedTerrans_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 315, 'general_ability_id': 251, 'goal': 'effect', 'name': 'Effect_InjectLarva_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 316, 'general_ability_id': 3747, 'goal': 'effect', 'name': 'Effect_InterferenceMatrix_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 317, 'general_ability_id': 2588, 'goal': 'effect', 'name': 'Effect_KD8Charge_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 538, 'general_ability_id': 2588, 'goal': 'effect', 'name': 'Effect_KD8Charge_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 318, 'general_ability_id': 2350, 'goal': 'effect', 'name': 'Effect_LockOn_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 541, 'general_ability_id': 2350, 'goal': 'effect', 'name': 'Effect_LockOn_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 319, 'general_ability_id': 2387, 'goal': 'effect', 'name': 'Effect_LocustSwoop_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 110, 'general_ability_id': 3686, 'goal': 'effect', 'name': 'Effect_MassRecall_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 320, 'general_ability_id': 2116, 'goal': 'effect', 'name': 'Effect_MedivacIgniteAfterburners_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 321, 'general_ability_id': 249, 'goal': 'effect', 'name': 'Effect_NeuralParasite_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 322, 'general_ability_id': 1622, 'goal': 'effect', 'name': 'Effect_NukeCalldown_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 90, 'general_ability_id': 2146, 'goal': 'effect', 'name': 'Effect_OracleRevelation_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 323, 'general_ability_id': 2542, 'goal': 'effect', 'name': 'Effect_ParasiticBomb_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 65, 'general_ability_id': 1036, 'goal': 'effect', 'name': 'Effect_PsiStorm_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 167, 'general_ability_id': 2346, 'goal': 'effect', 'name': 'Effect_PurificationNova_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 324, 'general_ability_id': 3685, 'goal': 'effect', 'name': 'Effect_Repair_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 108, 'general_ability_id': 3685, 'goal': 'effect', 'name': 'Effect_Repair_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 109, 'general_ability_id': 3685, 'goal': 'effect', 'name': 'Effect_Repair_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 331, 'general_ability_id': 3765, 'goal': 'effect', 'name': 'Effect_Restore_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 161, 'general_ability_id': 3765, 'goal': 'effect', 'name': 'Effect_Restore_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 332, 'general_ability_id': 32, 'goal': 'effect', 'name': 'Effect_Salvage_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 333, 'general_ability_id': 399, 'goal': 'effect', 'name': 'Effect_Scan_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 334, 'general_ability_id': 181, 'goal': 'effect', 'name': 'Effect_SpawnChangeling_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 335, 'general_ability_id': 2704, 'goal': 'effect', 'name': 'Effect_SpawnLocusts_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 336, 'general_ability_id': 2704, 'goal': 'effect', 'name': 'Effect_SpawnLocusts_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 337, 'general_ability_id': 3684, 'goal': 'effect', 'name': 'Effect_Spray_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 341, 'general_ability_id': 3675, 'goal': 'effect', 'name': 'Effect_Stim_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 346, 'general_ability_id': 255, 'goal': 'effect', 'name': 'Effect_SupplyDrop_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 347, 'general_ability_id': 2358, 'goal': 'effect', 'name': 'Effect_TacticalJump_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 348, 'general_ability_id': 2244, 'goal': 'effect', 'name': 'Effect_TimeWarp_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 349, 'general_ability_id': 1664, 'goal': 'effect', 'name': 'Effect_Transfusion_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 350, 'general_ability_id': 2073, 'goal': 'effect', 'name': 'Effect_ViperConsume_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 94, 'general_ability_id': 2393, 'goal': 'effect', 'name': 'Effect_VoidRayPrismaticAlignment_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 353, 'general_ability_id': 2099, 'goal': 'effect', 'name': 'Effect_WidowMineAttack_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 351, 'general_ability_id': 2099, 'goal': 'effect', 'name': 'Effect_WidowMineAttack_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 352, 'general_ability_id': 2099, 'goal': 'effect', 'name': 'Effect_WidowMineAttack_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 537, 'general_ability_id': 401, 'goal': 'effect', 'name': 'Effect_YamatoGun_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 93, 'general_ability_id': 2391, 'goal': 'effect', 'name': 'Hallucination_Adept_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 22, 'general_ability_id': 146, 'goal': 'effect', 'name': 'Hallucination_Archon_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 23, 'general_ability_id': 148, 'goal': 'effect', 'name': 'Hallucination_Colossus_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 92, 'general_ability_id': 2389, 'goal': 'effect', 'name': 'Hallucination_Disruptor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 24, 'general_ability_id': 150, 'goal': 'effect', 'name': 'Hallucination_HighTemplar_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 25, 'general_ability_id': 152, 'goal': 'effect', 'name': 'Hallucination_Immortal_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 89, 'general_ability_id': 2114, 'goal': 'effect', 'name': 'Hallucination_Oracle_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 26, 'general_ability_id': 154, 'goal': 'effect', 'name': 'Hallucination_Phoenix_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 27, 'general_ability_id': 156, 'goal': 'effect', 'name': 'Hallucination_Probe_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 28, 'general_ability_id': 158, 'goal': 'effect', 'name': 'Hallucination_Stalker_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 29, 'general_ability_id': 160, 'goal': 'effect', 'name': 'Hallucination_VoidRay_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 30, 'general_ability_id': 162, 'goal': 'effect', 'name': 'Hallucination_WarpPrism_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 31, 'general_ability_id': 164, 'goal': 'effect', 'name': 'Hallucination_Zealot_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 99, 'general_ability_id': 3660, 'goal': 'other', 'name': 'Halt_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 102, 'general_ability_id': 3666, 'goal': 'other', 'name': 'Harvest_Gather_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 103, 'general_ability_id': 3667, 'goal': 'other', 'name': 'Harvest_Return_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 17, 'general_ability_id': 3793, 'goal': 'other', 'name': 'HoldPosition_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 363, 'general_ability_id': 3678, 'goal': 'other', 'name': 'Land_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 369, 'general_ability_id': 3679, 'goal': 'other', 'name': 'Lift_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 375, 'general_ability_id': 3663, 'goal': 'other', 'name': 'LoadAll_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 104, 'general_ability_id': 3668, 'goal': 'other', 'name': 'Load_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 86, 'general_ability_id': 1766, 'goal': 'unit', 'name': 'Morph_Archon_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 141} ,
    {'func_id': 383, 'general_ability_id': 1372, 'goal': 'unit', 'name': 'Morph_BroodLord_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 114} ,
    {'func_id': 78, 'general_ability_id': 1520, 'goal': 'other', 'name': 'Morph_Gateway_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 384, 'general_ability_id': 1220, 'goal': 'build', 'name': 'Morph_GreaterSpire_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 102} ,
    {'func_id': 385, 'general_ability_id': 1998, 'goal': 'other', 'name': 'Morph_Hellbat_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 386, 'general_ability_id': 1978, 'goal': 'other', 'name': 'Morph_Hellion_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 387, 'general_ability_id': 1218, 'goal': 'build', 'name': 'Morph_Hive_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 101} ,
    {'func_id': 388, 'general_ability_id': 1216, 'goal': 'build', 'name': 'Morph_Lair_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 100} ,
    {'func_id': 389, 'general_ability_id': 2560, 'goal': 'other', 'name': 'Morph_LiberatorAAMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 390, 'general_ability_id': 2558, 'goal': 'other', 'name': 'Morph_LiberatorAGMode_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 392, 'general_ability_id': 2112, 'goal': 'build', 'name': 'Morph_LurkerDen_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 504} ,
    {'func_id': 391, 'general_ability_id': 2332, 'goal': 'unit', 'name': 'Morph_Lurker_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 502} ,
    {'func_id': 393, 'general_ability_id': 1847, 'goal': 'unit', 'name': 'Morph_Mothership_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 10} ,
    {'func_id': 121, 'general_ability_id': 3739, 'goal': 'other', 'name': 'Morph_ObserverMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 394, 'general_ability_id': 1516, 'goal': 'build', 'name': 'Morph_OrbitalCommand_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 132} ,
    {'func_id': 395, 'general_ability_id': 2708, 'goal': 'unit', 'name': 'Morph_OverlordTransport_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 893} ,
    {'func_id': 397, 'general_ability_id': 3745, 'goal': 'other', 'name': 'Morph_OverseerMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 396, 'general_ability_id': 1448, 'goal': 'unit', 'name': 'Morph_Overseer_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 129} ,
    {'func_id': 398, 'general_ability_id': 3743, 'goal': 'other', 'name': 'Morph_OversightMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 399, 'general_ability_id': 1450, 'goal': 'build', 'name': 'Morph_PlanetaryFortress_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 130} ,
    {'func_id': 400, 'general_ability_id': 2330, 'goal': 'unit', 'name': 'Morph_Ravager_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 688} ,
    {'func_id': 401, 'general_ability_id': 3680, 'goal': 'other', 'name': 'Morph_Root_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 402, 'general_ability_id': 388, 'goal': 'other', 'name': 'Morph_SiegeMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 407, 'general_ability_id': 556, 'goal': 'other', 'name': 'Morph_SupplyDepot_Lower_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 408, 'general_ability_id': 558, 'goal': 'other', 'name': 'Morph_SupplyDepot_Raise_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 160, 'general_ability_id': 3741, 'goal': 'other', 'name': 'Morph_SurveillanceMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 409, 'general_ability_id': 2364, 'goal': 'other', 'name': 'Morph_ThorExplosiveMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 410, 'general_ability_id': 2362, 'goal': 'other', 'name': 'Morph_ThorHighImpactMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 411, 'general_ability_id': 390, 'goal': 'other', 'name': 'Morph_Unsiege_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 412, 'general_ability_id': 3681, 'goal': 'other', 'name': 'Morph_Uproot_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 413, 'general_ability_id': 403, 'goal': 'other', 'name': 'Morph_VikingAssaultMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 414, 'general_ability_id': 405, 'goal': 'other', 'name': 'Morph_VikingFighterMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 77, 'general_ability_id': 1518, 'goal': 'other', 'name': 'Morph_WarpGate_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 544, 'general_ability_id': 1518, 'goal': 'other', 'name': 'Morph_WarpGate_autocast', 'queued': False, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 80, 'general_ability_id': 1528, 'goal': 'other', 'name': 'Morph_WarpPrismPhasingMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 81, 'general_ability_id': 1530, 'goal': 'other', 'name': 'Morph_WarpPrismTransportMode_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 13, 'general_ability_id': 3794, 'goal': 'other', 'name': 'Move_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 14, 'general_ability_id': 3794, 'goal': 'other', 'name': 'Move_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 15, 'general_ability_id': 3795, 'goal': 'other', 'name': 'Patrol_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 16, 'general_ability_id': 3795, 'goal': 'other', 'name': 'Patrol_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 106, 'general_ability_id': 3673, 'goal': 'other', 'name': 'Rally_Units_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 107, 'general_ability_id': 3673, 'goal': 'other', 'name': 'Rally_Units_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 114, 'general_ability_id': 3690, 'goal': 'other', 'name': 'Rally_Workers_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 115, 'general_ability_id': 3690, 'goal': 'other', 'name': 'Rally_Workers_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 425, 'general_ability_id': 3709, 'goal': 'research', 'name': 'Research_AdaptiveTalons_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 293} ,
    {'func_id': 85, 'general_ability_id': 1594, 'goal': 'research', 'name': 'Research_AdeptResonatingGlaives_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 130} ,
    {'func_id': 426, 'general_ability_id': 805, 'goal': 'research', 'name': 'Research_AdvancedBallistics_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 140} ,
    {'func_id': 553, 'general_ability_id': 263, 'goal': 'research', 'name': 'Research_AnabolicSynthesis_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 88} ,
    {'func_id': 427, 'general_ability_id': 790, 'goal': 'research', 'name': 'Research_BansheeCloakingField_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 20} ,
    {'func_id': 428, 'general_ability_id': 799, 'goal': 'research', 'name': 'Research_BansheeHyperflightRotors_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 136} ,
    {'func_id': 429, 'general_ability_id': 1532, 'goal': 'research', 'name': 'Research_BattlecruiserWeaponRefit_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 76} ,
    {'func_id': 84, 'general_ability_id': 1593, 'goal': 'research', 'name': 'Research_Blink_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 87} ,
    {'func_id': 430, 'general_ability_id': 1225, 'goal': 'research', 'name': 'Research_Burrow_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 64} ,
    {'func_id': 431, 'general_ability_id': 1482, 'goal': 'research', 'name': 'Research_CentrifugalHooks_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 75} ,
    {'func_id': 83, 'general_ability_id': 1592, 'goal': 'research', 'name': 'Research_Charge_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 86} ,
    {'func_id': 432, 'general_ability_id': 265, 'goal': 'research', 'name': 'Research_ChitinousPlating_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 4} ,
    {'func_id': 433, 'general_ability_id': 731, 'goal': 'research', 'name': 'Research_CombatShield_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 16} ,
    {'func_id': 434, 'general_ability_id': 732, 'goal': 'research', 'name': 'Research_ConcussiveShells_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 17} ,
    {'func_id': 554, 'general_ability_id': 769, 'goal': 'research', 'name': 'Research_CycloneLockOnDamage_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 144} ,
    {'func_id': 435, 'general_ability_id': 768, 'goal': 'research', 'name': 'Research_CycloneRapidFireLaunchers_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 291} ,
    {'func_id': 436, 'general_ability_id': 764, 'goal': 'research', 'name': 'Research_DrillingClaws_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 122} ,
    {'func_id': 563, 'general_ability_id': 822, 'goal': 'research', 'name': 'Research_EnhancedShockwaves_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 296} ,
    {'func_id': 69, 'general_ability_id': 1097, 'goal': 'research', 'name': 'Research_ExtendedThermalLance_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 50} ,
    {'func_id': 437, 'general_ability_id': 216, 'goal': 'research', 'name': 'Research_GlialRegeneration_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 2} ,
    {'func_id': 67, 'general_ability_id': 1093, 'goal': 'research', 'name': 'Research_GraviticBooster_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 48} ,
    {'func_id': 68, 'general_ability_id': 1094, 'goal': 'research', 'name': 'Research_GraviticDrive_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 49} ,
    {'func_id': 438, 'general_ability_id': 1282, 'goal': 'research', 'name': 'Research_GroovedSpines_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 134} ,
    {'func_id': 440, 'general_ability_id': 804, 'goal': 'research', 'name': 'Research_HighCapacityFuelTanks_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 139} ,
    {'func_id': 439, 'general_ability_id': 650, 'goal': 'research', 'name': 'Research_HiSecAutoTracking_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 5} ,
    {'func_id': 441, 'general_ability_id': 761, 'goal': 'research', 'name': 'Research_InfernalPreigniter_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 19} ,
    {'func_id': 18, 'general_ability_id': 44, 'goal': 'research', 'name': 'Research_InterceptorGravitonCatapult_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 1} ,
    {'func_id': 442, 'general_ability_id': 1283, 'goal': 'research', 'name': 'Research_MuscularAugments_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 135} ,
    {'func_id': 443, 'general_ability_id': 655, 'goal': 'research', 'name': 'Research_NeosteelFrame_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 10} ,
    {'func_id': 444, 'general_ability_id': 1455, 'goal': 'research', 'name': 'Research_NeuralParasite_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 101} ,
    {'func_id': 445, 'general_ability_id': 1454, 'goal': 'research', 'name': 'Research_PathogenGlands_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 74} ,
    {'func_id': 446, 'general_ability_id': 820, 'goal': 'research', 'name': 'Research_PersonalCloaking_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 25} ,
    {'func_id': 19, 'general_ability_id': 46, 'goal': 'research', 'name': 'Research_PhoenixAnionPulseCrystals_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 99} ,
    {'func_id': 447, 'general_ability_id': 1223, 'goal': 'research', 'name': 'Research_PneumatizedCarapace_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 62} ,
    {'func_id': 116, 'general_ability_id': 3692, 'goal': 'research', 'name': 'Research_ProtossAirArmor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 81} ,
    {'func_id': 117, 'general_ability_id': 3693, 'goal': 'research', 'name': 'Research_ProtossAirWeapons_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 78} ,
    {'func_id': 118, 'general_ability_id': 3694, 'goal': 'research', 'name': 'Research_ProtossGroundArmor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 42} ,
    {'func_id': 119, 'general_ability_id': 3695, 'goal': 'research', 'name': 'Research_ProtossGroundWeapons_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 39} ,
    {'func_id': 120, 'general_ability_id': 3696, 'goal': 'research', 'name': 'Research_ProtossShields_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 45} ,
    {'func_id': 70, 'general_ability_id': 1126, 'goal': 'research', 'name': 'Research_PsiStorm_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 52} ,
    {'func_id': 448, 'general_ability_id': 793, 'goal': 'research', 'name': 'Research_RavenCorvidReactor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 22} ,
    {'func_id': 449, 'general_ability_id': 803, 'goal': 'research', 'name': 'Research_RavenRecalibratedExplosives_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 97, 'general_ability_id': 2720, 'goal': 'research', 'name': 'Research_ShadowStrike_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 141} ,
    {'func_id': 450, 'general_ability_id': 766, 'goal': 'research', 'name': 'Research_SmartServos_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 289} ,
    {'func_id': 451, 'general_ability_id': 730, 'goal': 'research', 'name': 'Research_Stimpack_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 15} ,
    {'func_id': 452, 'general_ability_id': 3697, 'goal': 'research', 'name': 'Research_TerranInfantryArmor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 11} ,
    {'func_id': 456, 'general_ability_id': 3698, 'goal': 'research', 'name': 'Research_TerranInfantryWeapons_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 7} ,
    {'func_id': 460, 'general_ability_id': 3699, 'goal': 'research', 'name': 'Research_TerranShipWeapons_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 36} ,
    {'func_id': 464, 'general_ability_id': 651, 'goal': 'research', 'name': 'Research_TerranStructureArmorUpgrade_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 6} ,
    {'func_id': 465, 'general_ability_id': 3700, 'goal': 'research', 'name': 'Research_TerranVehicleAndShipPlating_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 116} ,
    {'func_id': 469, 'general_ability_id': 3701, 'goal': 'research', 'name': 'Research_TerranVehicleWeapons_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 30} ,
    {'func_id': 473, 'general_ability_id': 217, 'goal': 'research', 'name': 'Research_TunnelingClaws_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 3} ,
    {'func_id': 82, 'general_ability_id': 1568, 'goal': 'research', 'name': 'Research_WarpGate_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 84} ,
    {'func_id': 474, 'general_ability_id': 3702, 'goal': 'research', 'name': 'Research_ZergFlyerArmor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 71} ,
    {'func_id': 478, 'general_ability_id': 3703, 'goal': 'research', 'name': 'Research_ZergFlyerAttack_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 68} ,
    {'func_id': 482, 'general_ability_id': 3704, 'goal': 'research', 'name': 'Research_ZergGroundArmor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 56} ,
    {'func_id': 494, 'general_ability_id': 1252, 'goal': 'research', 'name': 'Research_ZerglingAdrenalGlands_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 65} ,
    {'func_id': 495, 'general_ability_id': 1253, 'goal': 'research', 'name': 'Research_ZerglingMetabolicBoost_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 66} ,
    {'func_id': 486, 'general_ability_id': 3705, 'goal': 'research', 'name': 'Research_ZergMeleeWeapons_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 53} ,
    {'func_id': 490, 'general_ability_id': 3706, 'goal': 'research', 'name': 'Research_ZergMissileWeapons_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 59} ,
    {'func_id': 1, 'general_ability_id': 1, 'goal': 'other', 'name': 'Smart_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 12, 'general_ability_id': 1, 'goal': 'other', 'name': 'Smart_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 101, 'general_ability_id': 3665, 'goal': 'other', 'name': 'Stop_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 54, 'general_ability_id': 922, 'goal': 'unit', 'name': 'Train_Adept_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 311} ,
    {'func_id': 498, 'general_ability_id': 80, 'goal': 'unit', 'name': 'Train_Baneling_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 9} ,
    {'func_id': 499, 'general_ability_id': 621, 'goal': 'unit', 'name': 'Train_Banshee_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 55} ,
    {'func_id': 500, 'general_ability_id': 623, 'goal': 'unit', 'name': 'Train_Battlecruiser_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 57} ,
    {'func_id': 56, 'general_ability_id': 948, 'goal': 'unit', 'name': 'Train_Carrier_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 79} ,
    {'func_id': 62, 'general_ability_id': 978, 'goal': 'unit', 'name': 'Train_Colossus_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 4} ,
    {'func_id': 501, 'general_ability_id': 1353, 'goal': 'unit', 'name': 'Train_Corruptor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 112} ,
    {'func_id': 502, 'general_ability_id': 597, 'goal': 'unit', 'name': 'Train_Cyclone_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 692} ,
    {'func_id': 52, 'general_ability_id': 920, 'goal': 'unit', 'name': 'Train_DarkTemplar_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 76} ,
    {'func_id': 166, 'general_ability_id': 994, 'goal': 'unit', 'name': 'Train_Disruptor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 694} ,
    {'func_id': 503, 'general_ability_id': 1342, 'goal': 'unit', 'name': 'Train_Drone_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 104} ,
    {'func_id': 504, 'general_ability_id': 562, 'goal': 'unit', 'name': 'Train_Ghost_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 50} ,
    {'func_id': 505, 'general_ability_id': 596, 'goal': 'unit', 'name': 'Train_Hellbat_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 484} ,
    {'func_id': 506, 'general_ability_id': 595, 'goal': 'unit', 'name': 'Train_Hellion_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 53} ,
    {'func_id': 51, 'general_ability_id': 919, 'goal': 'unit', 'name': 'Train_HighTemplar_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 75} ,
    {'func_id': 507, 'general_ability_id': 1345, 'goal': 'unit', 'name': 'Train_Hydralisk_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 107} ,
    {'func_id': 63, 'general_ability_id': 979, 'goal': 'unit', 'name': 'Train_Immortal_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 83} ,
    {'func_id': 508, 'general_ability_id': 1352, 'goal': 'unit', 'name': 'Train_Infestor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 111} ,
    {'func_id': 509, 'general_ability_id': 626, 'goal': 'unit', 'name': 'Train_Liberator_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 689} ,
    {'func_id': 510, 'general_ability_id': 563, 'goal': 'unit', 'name': 'Train_Marauder_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 51} ,
    {'func_id': 511, 'general_ability_id': 560, 'goal': 'unit', 'name': 'Train_Marine_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 48} ,
    {'func_id': 512, 'general_ability_id': 620, 'goal': 'unit', 'name': 'Train_Medivac_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 54} ,
    {'func_id': 513, 'general_ability_id': 1853, 'goal': 'unit', 'name': 'Train_MothershipCore_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 488} ,
    {'func_id': 21, 'general_ability_id': 110, 'goal': 'unit', 'name': 'Train_Mothership_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 10} ,
    {'func_id': 514, 'general_ability_id': 1346, 'goal': 'unit', 'name': 'Train_Mutalisk_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 108} ,
    {'func_id': 61, 'general_ability_id': 977, 'goal': 'unit', 'name': 'Train_Observer_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 82} ,
    {'func_id': 58, 'general_ability_id': 954, 'goal': 'unit', 'name': 'Train_Oracle_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 495} ,
    {'func_id': 515, 'general_ability_id': 1344, 'goal': 'unit', 'name': 'Train_Overlord_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 106} ,
    {'func_id': 55, 'general_ability_id': 946, 'goal': 'unit', 'name': 'Train_Phoenix_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 78} ,
    {'func_id': 64, 'general_ability_id': 1006, 'goal': 'unit', 'name': 'Train_Probe_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 84} ,
    {'func_id': 516, 'general_ability_id': 1632, 'goal': 'unit', 'name': 'Train_Queen_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 126} ,
    {'func_id': 517, 'general_ability_id': 622, 'goal': 'unit', 'name': 'Train_Raven_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 56} ,
    {'func_id': 518, 'general_ability_id': 561, 'goal': 'unit', 'name': 'Train_Reaper_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 49} ,
    {'func_id': 519, 'general_ability_id': 1351, 'goal': 'unit', 'name': 'Train_Roach_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 110} ,
    {'func_id': 520, 'general_ability_id': 524, 'goal': 'unit', 'name': 'Train_SCV_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 45} ,
    {'func_id': 53, 'general_ability_id': 921, 'goal': 'unit', 'name': 'Train_Sentry_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 77} ,
    {'func_id': 521, 'general_ability_id': 591, 'goal': 'unit', 'name': 'Train_SiegeTank_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 33} ,
    {'func_id': 50, 'general_ability_id': 917, 'goal': 'unit', 'name': 'Train_Stalker_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 74} ,
    {'func_id': 522, 'general_ability_id': 1356, 'goal': 'unit', 'name': 'Train_SwarmHost_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 494} ,
    {'func_id': 59, 'general_ability_id': 955, 'goal': 'unit', 'name': 'Train_Tempest_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 496} ,
    {'func_id': 523, 'general_ability_id': 594, 'goal': 'unit', 'name': 'Train_Thor_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 52} ,
    {'func_id': 524, 'general_ability_id': 1348, 'goal': 'unit', 'name': 'Train_Ultralisk_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 109} ,
    {'func_id': 525, 'general_ability_id': 624, 'goal': 'unit', 'name': 'Train_VikingFighter_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 35} ,
    {'func_id': 526, 'general_ability_id': 1354, 'goal': 'unit', 'name': 'Train_Viper_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 499} ,
    {'func_id': 57, 'general_ability_id': 950, 'goal': 'unit', 'name': 'Train_VoidRay_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 80} ,
    {'func_id': 76, 'general_ability_id': 1419, 'goal': 'unit', 'name': 'TrainWarp_Adept_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 311} ,
    {'func_id': 74, 'general_ability_id': 1417, 'goal': 'unit', 'name': 'TrainWarp_DarkTemplar_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 76} ,
    {'func_id': 73, 'general_ability_id': 1416, 'goal': 'unit', 'name': 'TrainWarp_HighTemplar_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 75} ,
    {'func_id': 60, 'general_ability_id': 976, 'goal': 'unit', 'name': 'Train_WarpPrism_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 81} ,
    {'func_id': 75, 'general_ability_id': 1418, 'goal': 'unit', 'name': 'TrainWarp_Sentry_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 77} ,
    {'func_id': 72, 'general_ability_id': 1414, 'goal': 'unit', 'name': 'TrainWarp_Stalker_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 74} ,
    {'func_id': 71, 'general_ability_id': 1413, 'goal': 'unit', 'name': 'TrainWarp_Zealot_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False, 'game_id': 73} ,
    {'func_id': 527, 'general_ability_id': 614, 'goal': 'unit', 'name': 'Train_WidowMine_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 498} ,
    {'func_id': 49, 'general_ability_id': 916, 'goal': 'unit', 'name': 'Train_Zealot_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 73} ,
    {'func_id': 528, 'general_ability_id': 1343, 'goal': 'unit', 'name': 'Train_Zergling_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False, 'game_id': 105} ,
    {'func_id': 105, 'general_ability_id': 3669, 'goal': 'other', 'name': 'UnloadAllAt_pt', 'queued': True, 'selected_units': True, 'target_location': True, 'target_unit': False} ,
    {'func_id': 164, 'general_ability_id': 3669, 'goal': 'other', 'name': 'UnloadAllAt_unit', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': True} ,
    {'func_id': 100, 'general_ability_id': 3664, 'goal': 'other', 'name': 'UnloadAll_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
    {'func_id': 556, 'general_ability_id': 3796, 'goal': 'other', 'name': 'UnloadUnit_quick', 'queued': True, 'selected_units': True, 'target_location': False, 'target_unit': False} ,
]

from distar.pysc2.lib.actions import RAW_FUNCTIONS
from collections import defaultdict
from distar.pysc2.lib.static_data import UNIT_SPECIFIC_ABILITIES, UNIT_GENERAL_ABILITIES, UNIT_MIX_ABILITIES, UNIT_TYPES, UPGRADES

NUM_UNIT_MIX_ABILITIES = len(UNIT_MIX_ABILITIES)  #269

UNIT_ABILITY_REORDER = torch.full((max(UNIT_MIX_ABILITIES) + 1, ), fill_value=-1, dtype=torch.long)
for idx in range(len(UNIT_SPECIFIC_ABILITIES)):
    if UNIT_GENERAL_ABILITIES[idx] == 0:
        UNIT_ABILITY_REORDER[UNIT_SPECIFIC_ABILITIES[idx]] = UNIT_MIX_ABILITIES.index(UNIT_SPECIFIC_ABILITIES[idx])
    else:
        UNIT_ABILITY_REORDER[UNIT_SPECIFIC_ABILITIES[idx]] = UNIT_MIX_ABILITIES.index(UNIT_GENERAL_ABILITIES[idx])
UNIT_ABILITY_REORDER[0] = 0 # use 0 as no op

FUNC_ID_TO_ACTION_TYPE_DICT = {a['func_id']: idx for idx, a in enumerate(ACTIONS)}
ABILITY_TO_GABILITY = {}
for idx in range(len(UNIT_SPECIFIC_ABILITIES)):
    if UNIT_GENERAL_ABILITIES[idx] == 0:
        ABILITY_TO_GABILITY[UNIT_SPECIFIC_ABILITIES[idx]] = UNIT_SPECIFIC_ABILITIES[idx]
    else:
        ABILITY_TO_GABILITY[UNIT_SPECIFIC_ABILITIES[idx]] = UNIT_GENERAL_ABILITIES[idx]

GABILITY_TO_QUEUE_ACTION = {}
QUEUE_ACTIONS = []
count = 1  # use 0 as no op
for idx, f in enumerate(ACTIONS):
    if 'Train_' in f['name'] or 'Research' in f['name']:
        GABILITY_TO_QUEUE_ACTION[f['general_ability_id']] = count
        QUEUE_ACTIONS.append(idx)
        count += 1
    else:
        GABILITY_TO_QUEUE_ACTION[f['general_ability_id']] = 0

ABILITY_TO_QUEUE_ACTION = torch.full((max(ABILITY_TO_GABILITY.keys()) + 1, ), fill_value=-1, dtype=torch.long)
ABILITY_TO_QUEUE_ACTION[0] = 0  # use 0 as no op
for a_id, g_id in ABILITY_TO_GABILITY.items():
    if g_id in GABILITY_TO_QUEUE_ACTION.keys():
        ABILITY_TO_QUEUE_ACTION[a_id] = GABILITY_TO_QUEUE_ACTION[g_id]
    else:
        ABILITY_TO_QUEUE_ACTION[a_id] = 0

EXCLUDE_ACTIONS = [
    'Build_Pylon_pt', 'Train_Overlord_quick', 'Build_SupplyDepot_pt',  # supply action
    'Train_Drone_quick', 'Train_SCV_quick', 'Train_Probe_quick',  # worker action
    'Build_CreepTumor_pt', ''
]

BO_EXCLUDE_ACTIONS = [
]

CUM_EXCLUDE_ACTIONS = [
    'Build_SpineCrawler_pt', 'Build_SporeCrawler_pt', 'Build_PhotonCannon_pt', 'Build_ShieldBattery_pt',
    'Build_Bunker_pt', 'Morph_Overseer_quick', 'Build_MissileTurret_pt'
]

BEGINNING_ORDER_ACTIONS = [0]
CUMULATIVE_STAT_ACTIONS = [0]
for idx, f in enumerate(ACTIONS):
    if f['goal'] in ['unit', 'build', 'research'] and f['name'] not in EXCLUDE_ACTIONS and f['name'] not in CUM_EXCLUDE_ACTIONS:
        CUMULATIVE_STAT_ACTIONS.append(idx)
    if f['goal'] in ['unit', 'build', 'research'] and f['name'] not in EXCLUDE_ACTIONS:
        BEGINNING_ORDER_ACTIONS.append(idx)

NUM_ACTIONS = len(ACTIONS)
NUM_QUEUE_ACTIONS = len(QUEUE_ACTIONS)
NUM_BEGINNING_ORDER_ACTIONS = len(BEGINNING_ORDER_ACTIONS)
NUM_CUMULATIVE_STAT_ACTIONS = len(CUMULATIVE_STAT_ACTIONS)

SELECTED_UNITS_MASK = torch.zeros(len(ACTIONS), dtype=torch.bool)
for idx, a in enumerate(ACTIONS):
    if a['selected_units']:
        SELECTED_UNITS_MASK[idx] = 1

UNIT_BUILD_ACTIONS = [a['func_id'] for a in ACTIONS if a['goal'] == 'build']
UNIT_TRAIN_ACTIONS = [a['func_id'] for a in ACTIONS if a['goal'] == 'unit']

GENERAL_ABILITY_IDS = []
for idx, a in enumerate(ACTIONS):
    GENERAL_ABILITY_IDS.append(a['general_ability_id'])
UNIT_ABILITY_TO_ACTION = {}
for idx, a in enumerate(UNIT_MIX_ABILITIES):
    if a in GENERAL_ABILITY_IDS:
        UNIT_ABILITY_TO_ACTION[idx] = GENERAL_ABILITY_IDS.index(a)

UNIT_TO_CUM = defaultdict(lambda: -1)
UPGRADE_TO_CUM = defaultdict(lambda: -1)
for idx, a in enumerate(ACTIONS):
    if 'game_id' in a and a['goal'] in ['unit', 'build'] and idx in CUMULATIVE_STAT_ACTIONS:
        UNIT_TO_CUM[a['game_id']] = CUMULATIVE_STAT_ACTIONS.index(idx)
    elif 'game_id' in a and a['goal'] in ['research'] and idx in CUMULATIVE_STAT_ACTIONS:
        UPGRADE_TO_CUM[a['game_id']] = CUMULATIVE_STAT_ACTIONS.index(idx)

