# R5 — Atmospheric interpretation and candidate-predictor review

## Why 850 hPa is defensible, and where it is not

The 850-hPa fields are retained as **lower-tropospheric context**, not as a direct representation of rain at a gauge. For northern Thailand, published monsoon work uses 850-hPa winds to describe low-level seasonal circulation, and Indo-China research explicitly links 850-hPa flow with wind–terrain–rainfall interaction. This is physically relevant to the Upper Ping basin because moisture advection and flow direction relative to mountain barriers can condition convection and orographic enhancement.

The limitation is important. The 40 stations span about 269–1,180 m elevation, while ERA5 is a coarse grid and its model orography cannot resolve individual valleys and ridges. At high stations the 850-hPa surface is comparatively close to the terrain; values should therefore be conditioned on the existing `valid_850` flag and interpreted as grid-scale environmental flow/moisture, never as a station boundary-layer measurement. The current canonical already carries 925-hPa fields and `valid_925`; these may be even more terrain-sensitive and should not be substituted blindly.

Evidence: [ERA5 pressure levels and parameter documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=669811810), [Indo-China monsoon indices and terrain interaction](https://pmc.ncbi.nlm.nih.gov/articles/PMC4309958/), and [Thailand monsoon onset study using Chiang Mai 850-hPa winds](https://doi.org/10.1002/joc.8862).

## What is available now

The project’s read-only ERA5 station-hour canonical contains `d2m`, `u10`, `v10`, `sp`, `q_850`, `u_850`, `v_850`, `q_925`, `u_925`, `v_925`, `valid_850`, and `valid_925` for 2013–2025. It does **not** contain CAPE, total-column water vapour, vertical velocity, cloud liquid/ice water, or cloud cover. Adding any candidate below therefore requires a new, separately versioned extraction and must not alter the frozen model-ready dataset.

## Ranked candidates

| Rank | Candidate | ERA5 availability | Expected value | Main caveat / test |
|---|---|---|---|---|
| 1 | Total column water vapour (TCWV) | Single-level / vertically integrated field | Measures column moisture supply beyond one pressure level; plausible discriminator between isolated heavy convection and moisture-rich widespread rain | Coarse grid; high TCWV is not sufficient for convection. Test only on development/validation periods. |
| 2 | CAPE | Single-level diagnostic | Adds convective instability, complementary to moisture and low-level wind | Forecast-derived/diagnostic behavior and timing must be documented; CAPE can be noisy and zero-inflated. Use lagged values available before target time. |
| 3 | Vertical velocity (`omega`) at 850/700/500 hPa | Pressure-level field | Represents ascent and vertical structure; 700/500 hPa reduce near-terrain ambiguity | Reanalysis ascent is grid-scale and may miss localized convection; sign convention must be explicit. |
| 4 | Total column cloud liquid water / ice water | Single-level total-column fields | May separate cloud-bearing systems and sustained stratiform/heavy-tail regimes | Strongly model-dependent; possible simultaneity/leakage if sampled too near the target hour. |
| 5 | Cloud liquid/ice specific water and relative humidity profiles | Pressure-level fields | Adds vertical cloud/moisture structure beyond `q_850` | More dimensions, collinearity, and pressure surfaces below/near terrain. Pre-register levels and missing rules. |
| 6 | Low/mid/high/total cloud cover | Single-level diagnostics | Simple cloud-regime indicators and useful for false-positive analysis | Cloud fraction is broad and may not resolve convective cores. |

ERA5 officially provides surface/single-level, vertical-integral, and pressure-level parameter families; the pressure-level archive includes 850 hPa among its standard levels. Cloud liquid/ice and total-column cloud diagnostics are defined in ECMWF documentation. Sources: [Copernicus ERA5 single-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=documentation), [ERA5 data documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=669811810), [ECMWF IFS cloud diagnostics](https://confluence.ecmwf.int/download/attachments/19661682/IFS_CY38R1_Part4.pdf?api=v2&download=true&modificationDate=1443006083445&version=1).

## Scientific decision

No new predictor is promoted into the final model in this revision. The defensible next experiment is a pre-registered development-only ablation beginning with TCWV, CAPE, and multi-level omega, with 2025 kept untouched as final holdout. Cloud variables are second-line candidates for regime interpretation and false-positive analysis.
