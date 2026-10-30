"""Current CCDC display names; historical data identifiers remain unchanged."""

METHOD_NAME = 'CaMRA'
METHOD_FULL_NAME = 'Calibrated Minimum-Cost Reach-Avoid'
METHOD_ALIASES = {'HD-MCRA': METHOD_NAME, 'hd_mcra': METHOD_NAME}


def display_name(identifier):
    return METHOD_ALIASES.get(identifier, identifier)
