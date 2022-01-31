import importlib
import warnings
from typing import List

global ceph_flag, mc_flag, linklink_flag, redis_flag
ceph_flag, mc_flag, linklink_flag,redis_flag = True, True, True, True

def try_import_redis():
    """
    Overview:
        Try import ceph module, if failed, return None

    Returns:
        module: imported module, or None when ceph not found
    """
    global redis_flag
    try:
        import redis
        from redis.client import StrictRedis
        return redis,StrictRedis
    except ModuleNotFoundError as e:
        if redis_flag:
            warnings.warn(
                "You have not installed redis package!",
            )
        redis_flag = False
        return None, None


def try_import_ceph():
    """
    Overview:
        Try import ceph module, if failed, return None

    Returns:
        module: imported module, or None when ceph not found
    """
    global ceph_flag
    return None
    # try:
    #     import ceph
    #     client = ceph.S3Client()
    #     return client
    # except ModuleNotFoundError as e:
    #     try:
    #         from petrel_client.client import Client
    #         client = Client(conf_path='~/petreloss.conf')
    #         return client
    #     except ModuleNotFoundError as e:
    #         if ceph_flag:
    #             warnings.warn(
    #                 "You have not installed ceph package! If you are not run locally and testing, "
    #                 "ask coworker for help."
    #             )
    #         ceph = None
    #         ceph_flag = False
    #         return ceph


def try_import_mc():
    """
    Overview:
        Try import mc module, if failed, return None

    Returns:
        module: imported module, or None when mc not found
    """
    global mc_flag
    mc_flag
    mc = None
    return mc
    try:
        import mc
    except ModuleNotFoundError as e:
        if mc_flag:
            warnings.warn(
                "You have not installed mc package! If you are not run locally and testing, "
                "ask coworker for help."
            )
        mc = None
        mc_flag = False
    return mc


def try_import_link():
    global linklink_flag
    """
    Overview:
        Try import linklink module, if failed, import nervex.tests.fake_linklink instead

    Returns:
        module: imported module (may be fake_linklink)
    """
    try:
        import torch.distributed as link
    except ModuleNotFoundError as e:
        if linklink_flag:
            warnings.warn(
                "You have not installed linklink package! If you are not run locally and testing, "
                "ask coworker for help. We will run a fake linklink."
                "Refer to nervex.utils.fake_linklink.py for details."
            )
        from distar.ctools.utils import link
        linklink_flag = False
    return link


def import_module(modules: List[str]) -> None:
    """
    Overview:
        Import several module as a list
    Args:
        - modules (:obj:`list` of `str`): List of module names
    """
    for name in modules:
        importlib.import_module(name)
