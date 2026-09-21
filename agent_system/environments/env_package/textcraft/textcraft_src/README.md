# Vendored TextCraft (for textcraft_ood)

- `textcraft/{crafting_tree.py, env.py, utils.py}`: copied verbatim from the official TextCraft
  environment in ADaPT (https://github.com/archiki/ADaPT, commit `ecdc4ab`, directory `TextCraft/`).
  MIT License, Copyright (c) 2023 Archiki Prasad; see `LICENSE` in this directory.
- `textcraft/recipes/`: 860 Minecraft 1.16.5 crafting recipes, taken from the source ADaPT's README
  names (https://github.com/InventivetalentDev/minecraft-assets/tree/1.16.5/data/minecraft/recipes).
  `RECIPES_MANIFEST.sha256` lists every file's checksum; the upstream files are byte-identical.
- `../textcraft_pkg` is a relative symlink to `textcraft/`. The vendored modules import each other as
  `textcraft.*`, so the package directory must keep that name; `envs.py` puts this directory's parent
  on `sys.path`.
