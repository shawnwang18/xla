"""Provides the repository macro to import raft."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports raft."""

    RAFT_COMMIT = "f54ba714aa42732413350e4c2a46639ab3b67ad5"
    RAFT_SHA256 = "bfd6818920536c9e76937d3d1e37f2d112bdbdbe589ffd29c6b687609e3cd999"

    tf_http_archive(
        name = "raft",
        sha256 = RAFT_SHA256,
        strip_prefix = "raft-{commit}".format(commit = RAFT_COMMIT),
        urls = tf_mirror_urls("https://github.com/rapidsai/raft/archive/{commit}.tar.gz".format(commit = RAFT_COMMIT)),
        build_file = "//third_party/raft:raft.BUILD",
        patch_file = [
            "//third_party/raft:clang_cuda_intrinsics.h.patch",
            "//third_party/raft:logger_macros.hpp.patch",
            "//third_party/raft:select_k_runner.hpp.patch",
            "//third_party/raft:select_k_runner.cu.cc.patch",
            "//third_party/raft:select_k_smoke_test.cu.cc.patch",
        ],
    )
