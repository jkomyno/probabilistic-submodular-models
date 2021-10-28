def float_to_str(x: float) -> str:
    """
    Convert a float to a string without scientific notation.
    It might have slight rounding errors in cases such as 0.3050500005,
    where the last non-zero digit is preceded by a large number of zeros
    and other non-zero digit before the decimal point.

    Examples:
    - 1.0 -> '1'
    - 10.0 -> '10'
    - 0.001 -> '0.001'
    - 0.0000001 -> '0.0000001' (rather than 1e-07)
    - 0.3050500005 -> '0.30505' (slight rounding error)
    """
    x_as_str = f'{x:g}'

    if x_as_str[:2] == '1e':
        x_as_str = f'{x:.15f}'.rstrip('0')

    return x_as_str
