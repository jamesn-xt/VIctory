# RTX 4080 scene and material experiments

This branch builds on `test/rtx4080-scene-completeness` (4ce2546). It combines community fixes for shader refusals, missing passes, stale animated textures and input. It is an experimental integration, not a claim of complete rendering or a measured frame-rate improvement. Original authors are retained in commits.

| Community change | Intended effect |
| --- | --- |
| [#52](https://github.com/fxceless/victory-gta6/pull/52) | Retain live first-half interpolation values; our preceding run rejected pixel shaders for this exact unsupported instruction. |
| [#40](https://github.com/fxceless/victory-gta6/pull/40) | Lower row permutations on wave32 hosts, retaining rejection of unsupported divergent inactive-lane fetches. |
| [#9](https://github.com/fxceless/victory-gta6/pull/9) | Support double-to-float conversion and whole-wave double sums for missing compute programs. |
| [#36](https://github.com/fxceless/victory-gta6/pull/36) | Recognize uniform LDS reads and exact operand widths without treating per-lane reads as uniform. |
| [#63](https://github.com/fxceless/victory-gta6/pull/63) | Refresh recently changing textures at every check; targets stale startup video frames and flicker. |
| [#74](https://github.com/fxceless/victory-gta6/pull/74) | Merge keyboard controls into the first connected pad, restoring physical axes when keys are released. |
| [#73](https://github.com/fxceless/victory-gta6/pull/73) | Emit PSV0 semantic strings per element for Windows shader-validator compatibility. |
| [#51](https://github.com/fxceless/victory-gta6/pull/51) | Disable blending on integer targets so D3D12 accepts the pipeline. Original fix: pablogonz12, [upstream #806](https://github.com/Marvisak/VIctory/pull/806); fork port/testing: pgm1207. |
| [#32](https://github.com/fxceless/victory-gta6/pull/32) | Use an FP16 render-target stand-in where the host cannot render to RGB9E5. Existing stand-in copy limitations remain. |
| [#35](https://github.com/fxceless/victory-gta6/pull/35) | Separate cached translations by effective host wave width; prevents reuse of incompatible adapter outputs. |

The prior branch already includes uniform sampler choices (#25), descriptor-only size queries (upstream #1067), and longer input history (#23).

## Build and launch

Use the repository's normal Rust/MSVC/DXC setup and a local `victory.toml` pointing at your own files, cache and data directory:

```powershell
cargo build --profile diag -p victory --features diagnostics --locked
$env:VICTORY_RELAY = '2'
$env:VICTORY_LATE_WAIT = '2'
$env:VICTORY_WIDE_BACK = '1'
$env:VICTORY_CARD_BUFFERS_MIN_MB = '1'
$env:VICTORY_CARD_BUFFERS_MB = '1024'
$env:VICTORY_WIDE_REFRESH = '1'
& .\target\diag\victory.exe run '<local game executable>' --execute --console series-x
```

`LATE_WAIT=2` retains submission overlap but waits at presentation; this is a presentation experiment, not proof that asynchronous presentation caused flicker. `LATE_WAIT=3` restores the preceding run's asynchronous presentation setting for comparison. `WIDE_REFRESH=1` enables existing parallel host-copy refresh code; its speed benefit has not been measured on this machine. These settings do not enable the incomplete BVH builder or wide-wave mode, change loop limits, or modify Windows TDR.

The first run after shader changes can spend substantial time rebuilding the cache. Loading-screen FPS is not gameplay FPS. Assess normal final output, material completeness, input response and sustained world FPS separately. No proprietary captures, caches, binaries or saves are included in this branch.
