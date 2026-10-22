# mudarkroom

The darkroom for the master universe project: code by Raellabot that develops Hubble and Webb mosaics from MAST into color pictures.

The code here is Raellabot's. The pictures are made from public-domain Hubble and Webb data (NASA/ESA/CSA/STScI) downloaded from MAST; this is an independent project, not made or endorsed by NASA, ESA or STScI.

## Setup

Put your own name and contact URL in the `UA` line at the top of `scripts/mast/color.py` and `scripts/mast/find_rows.py`, or set the `CONTACT_INFO` environment variable (a repository secret works in Actions); the variable wins when it is set. It is sent to MAST as the User-Agent so STScI can reach whoever is calling.

## Scripts

- `scripts/mast/find_rows.py OUTDIR COLLECTION PROPOSAL_ID [TARGET]` looks up a program's level-3 mosaics in MAST.
- `scripts/mast/probe_sizes.py OUTDIR COLLECTION|PROPOSAL ...` lists each field's filters and file sizes without downloading anything.
- `scripts/mast/color.py OUTDIR ROWS_JSON_GZ SPEC ...` downloads the filters of one field, aligns them by sky coordinates and colors them by wavelength order.

A SPEC is `COLLECTION|target_name|proposal_id|instrument_name|label[|black_percentile]`, for example `JWST|SMACS-J0723.3-7327|2736|NIRCAM/IMAGE|smacs0723|50`.

## Settings for color.py (environment variables)

| Name | Meaning |
| --- | --- |
| `MAX_SIDE` | Longest side of the picture in pixels (default 2600). |
| `MAX_FILTERS` | Most filters in one picture (default 6); more are thinned to an even spread of wavelengths. |
| `WORK_DIR` | Where downloads and the aligned arrays are kept (default `work`); FITS files are deleted after use and the arrays stay, so a rerun that only changes color or stretch skips the downloads. |
| `STRETCH`, `Q` | Lupton asinh stretch and softening (defaults 0.25 and 8). |
| `PALETTES` | Comma-separated palettes (chromatic, creative, creative2, creative3, stsci, creativerank, creative5). |
| `SIZE_CAP_GB` | Skip a filter whose file is larger than this (default 6). |
| `FLATTEN` | 1 removes the slowly changing sky background of each filter (old palettes only); 0 leaves it. |
| `NO_DOWNLOAD` | 1 uses only the cached aligned arrays and skips a set without them. |
| `NORM_EXPS` | Brightness exponents for the ranked palettes: 1 scales each filter on its own, 0 puts all on one scale. |
| `VARIANTS` | Tweaks of the ranked palettes: base, deep, soft, deepsoft, white, whiter, gentle, gentler, export. |
| `FINISH` | 1 (default) applies the per-subject finishing steps in the `FINISH` table of the script. |

Workflows (run by hand): `Color pictures` and `Probe sizes`. Results go to the `results` branch.
