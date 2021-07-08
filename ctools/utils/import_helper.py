import importlib
import warnings
from typing import List

global ceph_flag, mc_flag, linklink_flag
ceph_flag, mc_flag, linklink_flag = True, True, True


def try_import_ceph():
    """
    Overview:
        Try import ceph module, if failed, return None

    Returns:
        module: imported module, or None when ceph not found
    """
    global ceph_flag
    return None
    try:
        import ceph
        client = ceph.S3Client()
        return client
    except ModuleNotFoundError as e:
        try:
            from petrel_client.client import Client
            client = Client(conf_path='~/petreloss.conf')
            return client
        except ModuleNotFoundError as e:
            ceph = None
            ceph_flag = False
            return ceph


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
        Try import linklink module, if failed, import ctools.tests.fake_linklink instead

    Returns:
        module: imported module (may be fake_linklink)
    """
    try:
        import linklink as link
    except ModuleNotFoundError as e:
        from ctools.utils import link
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
