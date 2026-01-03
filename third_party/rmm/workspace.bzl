"""Provides the repository macro to import rmm."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports rmm."""

    RMM_COMMIT = "fc5e45bde2b7c543f902455788b6d74cc389c6c7"
    RMM_SHA256 = "c6e6a8fe9854ec32dc9381f2abeeb21aa7a540746b029dd000cd3f55c5f99d17"

    tf_http_archive(
        name = "rmm",
        sha256 = RMM_SHA256,
        strip_prefix = "rmm-{commit}".format(commit = RMM_COMMIT),
        urls = tf_mirror_urls("https://github.com/rapidsai/rmm/archive/{commit}.tar.gz".format(commit = RMM_COMMIT)),
        build_file = "//third_party/rmm:rmm.BUILD",
        patch_file = ["//third_party/rmm:logger_macros.hpp.patch"],
    )
