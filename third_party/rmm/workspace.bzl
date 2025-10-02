"""Provides the repository macro to import rmm."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports rmm."""

    RMM_VERSION = "25.12.00a"
    RMM_SHA256 = "6b1b37d9c4cf52dd47f3d59c86179e2b1f47413d65250b2ecd934dcffdfb2d2b"

    tf_http_archive(
        name = "rmm",
        sha256 = RMM_SHA256,
        strip_prefix = "rmm-{version}".format(version = RMM_VERSION),
        urls = tf_mirror_urls("https://github.com/rapidsai/rmm/archive/refs/tags/v{version}.tar.gz".format(version = RMM_VERSION)),
        build_file = "//third_party/rmm:rmm.BUILD",
        patch_file = ["//third_party/rmm:logger_macros.hpp.patch"],
    )
