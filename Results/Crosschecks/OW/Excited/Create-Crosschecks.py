import re
from pathlib import Path

input_file = Path("CC-OW.txt")
header_re = re.compile(r"^(\d+)\s+(Bs|DsStar).*")
# stricter float regex
values_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")

delta_re = re.compile(r"DeltaEx pSq=(\d+):\s+([0-9.]+)\((\d+)\)")

outfiles = {}
current_bs_vals = None
current_bs_delta = None
current_type = None
current_id = None
current_dsstar_key = None

with input_file.open() as f:
    for line in f:
        header_match = header_re.match(line.strip())
        if header_match:
            current_id, current_type = header_match.groups()
            if current_type == "Bs":
                current_bs_vals = []
                current_bs_delta = None
            elif current_type == "DsStar":
                amc_match = re.search(r"amx=([\d\.]+)", line)
                psq_match = re.search(r"pSq=(\d+)", line)
                if not (amc_match and psq_match and current_bs_vals):
                    continue
                amc = float(amc_match.group(1))
                psq = int(psq_match.group(1))
                key = f"{current_id}-amc{amc:.3f}.txt"

                if key not in outfiles:
                    outfiles[key] = {"Bs": current_bs_vals.copy(),
                                     "BsDelta": current_bs_delta,
                                     "DsStar": {},
                                     "DsStarDelta": {}}
                outfiles[key]["DsStar"][psq] = []
                current_dsstar_key = key, psq
            continue

        # normal value rows
        nums = values_re.findall(line)
        if len(nums) >= 8:
            vals = list(map(float, nums[:8]))
            if current_type == "Bs":
                current_bs_vals.append(vals)
            elif current_type == "DsStar" and current_dsstar_key:
                key, psq = current_dsstar_key
                outfiles[key]["DsStar"][psq].append(vals)

        # DeltaEx rows
        delta_match = delta_re.search(line)
        if delta_match:
            psq = int(delta_match.group(1))
            val_str, err_str = delta_match.group(2), delta_match.group(3)

            # scale error correctly
            if "." in val_str:
                decimals = len(val_str.split(".")[1])
            else:
                decimals = 0
            scale = 10 ** -decimals
            central = float(val_str)
            error = int(err_str) * scale

            if current_type == "Bs":
                current_bs_delta = (central, error)
            elif current_type == "DsStar" and current_dsstar_key:
                key, psq = current_dsstar_key
                outfiles[key]["DsStarDelta"][psq] = (central, error)


for key, blocks in outfiles.items():
    with open(key, "w") as out:
        out.write("index value uncertainty\n")
        if not blocks["Bs"]:
            continue

        selected_pairs = [0, 2]
        index = 0
        for pair in selected_pairs:
            v_bs = blocks["Bs"][0][2*pair]
            dv_bs = blocks["Bs"][0][2*pair+1]
            out.write(f"{index} {v_bs:.10e} {dv_bs:.10e}\n")
            index += 1

            for psq in sorted(blocks["DsStar"].keys()):
                if not blocks["DsStar"][psq]:
                    continue
                v_ds = blocks["DsStar"][psq][0][2*pair]
                dv_ds = blocks["DsStar"][psq][0][2*pair+1]
                out.write(f"{index} {v_ds:.10e} {dv_ds:.10e}\n")
                index += 1

        # --- DeltaEx block ---
        if blocks.get("BsDelta"):
            v_bs, dv_bs = blocks["BsDelta"]
            out.write(f"{index} {v_bs:.10e} {dv_bs:.10e}\n")
            index += 1

        for psq in sorted(blocks["DsStarDelta"].keys()):
            v_ds, dv_ds = blocks["DsStarDelta"][psq]
            out.write(f"{index} {v_ds:.10e} {dv_ds:.10e}\n")
            index += 1
