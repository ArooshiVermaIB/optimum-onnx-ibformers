from packaging import version
import os
import logging


FUTURE_VERSION = version.parse("99.99.99")
TABLE_FORMAT_MINIMAL_VERSION = version.parse("22.03.0")


def get_current_ib_version():
    if "RELEASE_VERSION" in os.environ:
        return version.parse(os.environ["RELEASE_VERSION"])
    if "RELEASE_BRANCH" in os.environ:
        raw_version = os.environ["RELEASE_BRANCH"].replace("release-", "")
        return version.parse(raw_version)
    else:
        logging.warning(
            "Could not work out the current version of instabase. Assuming newest possible for any "
            f"version-related decisions"
        )
        return FUTURE_VERSION


CURRENT_IB_VERSION = get_current_ib_version()
SHOULD_FORMAT = CURRENT_IB_VERSION >= TABLE_FORMAT_MINIMAL_VERSION
