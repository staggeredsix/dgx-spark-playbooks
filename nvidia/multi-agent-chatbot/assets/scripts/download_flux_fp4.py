from huggingface_hub import snapshot_download


def main() -> None:
    download_path = snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev-onnx",
        allow_patterns=["transformer.opt/fp4/**"],
        local_dir="flux-fp4",
        local_dir_use_symlinks=False,
    )
    print(download_path)


if __name__ == "__main__":
    main()
