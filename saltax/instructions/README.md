# Instructions

## Scope

Generator for simulation instructions in `fuse` format.

## Structure

- `generator.py`: `fuse` instruction infrastructures and instruction generators.

## Available Generators

- `generator_flat`: Flat spectrum of a certain NEST type.
- `generator_se`: Vanilla single electron peaks generator, which is data-independent.
- `generator_se_bootstrapped`: We will use XYT information from bootstrapped data single electrons to make the simulation more realistic. You have to specify a `xyt_files_at` to get it work, if you are not on Midway.
- `generator_ambe`: AmBe neutron simulator based on full-chain simulation instruction (post-`epix`). You have to specify instruction files `ambe_instructions_file` to make it work, if you are not on Midway.
