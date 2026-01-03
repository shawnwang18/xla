"""Provides the repository macro to import rmm."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports rmm."""

    RMM_VERSION = "25.12.00"
    RMM_SHA256 = "730b9a52bb83e87866a2f9e70a63857e47d010f35c18187e04e121d6e92436e4"

    tf_http_archive(
        name = "rmm",
        sha256 = RMM_SHA256,
        strip_prefix = "rmm-{version}".format(version = RMM_VERSION),
        urls = tf_mirror_urls("https://github.com/rapidsai/rmm/archive/refs/tags/v{version}.tar.gz".format(version = RMM_VERSION)),
        build_file = "//third_party/rmm:rmm.BUILD",
        patch_file = ["//third_party/rmm:logger_macros.hpp.patch"],
    )
