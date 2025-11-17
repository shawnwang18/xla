"""Provides the repository macro to import rmm."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports rmm."""

    RMM_COMMIT = "889309b691ae8e5c0491ea4598c43fdeaff3c09a"
    RMM_SHA256 = "1dbe69f38d58abd6375b2b106d4bc3ba425e51cf5cf43430f2a7e1fd11179398"

    tf_http_archive(
        name = "rmm",
        sha256 = RMM_SHA256,
        strip_prefix = "rmm-{commit}".format(commit = RMM_COMMIT),
        urls = tf_mirror_urls("https://github.com/rapidsai/rmm/archive/{commit}.tar.gz".format(commit = RMM_COMMIT)),
        build_file = "//third_party/rmm:rmm.BUILD",
        patch_file = ["//third_party/rmm:logger_macros.hpp.patch"],
    )
