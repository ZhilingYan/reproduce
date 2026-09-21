# Vendored TextCraft (for textcraft_ood)

- `textcraft/{crafting_tree.py, env.py, utils.py}`: copied verbatim from the official TextCraft
  environment in ADaPT (https://github.com/archiki/ADaPT, commit `ecdc4ab`, directory `TextCraft/`).
  MIT License, Copyright (c) 2023 Archiki Prasad; see `LICENSE` in this directory.
- `textcraft/recipes/` is **not** in this repository. It holds 860 Minecraft 1.16.5 crafting recipes,
  which ADaPT also does not redistribute (its README points to
  https://github.com/InventivetalentDev/minecraft-assets/tree/1.16.5/data/minecraft/recipes).
  Fetch them with `bash examples/data_preprocess/fetch_textcraft_recipes.sh`, which copies exactly the
  files listed in `RECIPES_MANIFEST.sha256` and verifies each checksum.
- `../textcraft_pkg` is a relative symlink to `textcraft/`. The vendored modules import each other as
  `textcraft.*`, so the package directory must keep that name; `envs.py` puts this directory's parent
  on `sys.path`.
