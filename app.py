def _parse_hkl(hkl_label: str) -> tuple:
    """Parse '(311)' → (3,1,1) or '(3,1,1)' → (3,1,1)"""
    clean = hkl_label.strip().strip("()").replace(" ", "")
    if "," in clean:
        return tuple(int(p.strip()) for p in clean.split(","))
    # Concatenated format: parse digit-by-digit with sign handling
    result, i = [], 0
    while i < len(clean) and len(result) < 3:
        sign = -1 if clean[i] == '-' else 1
        if clean[i] in "+-": i += 1
        num = ""
        while i < len(clean) and clean[i].isdigit():
            num += clean[i]; i += 1
        if num: result.append(sign * int(num))
    return tuple(result + [0]*(3-len(result)))  # Pad with zeros
