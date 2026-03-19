# src/decision_engine.py

def decide_what_to_do(predicted_state, intensity, stress_level, energy_level):
    """
    Rule-based wellness recommendation engine.
    Priority order:
      1. Override rules (extreme cases)
      2. State-based rules modified by stress/energy
    """

    # ── OVERRIDE RULES (highest priority) ──
    # Very low energy → rest no matter what
    if energy_level <= 1:
        return 'rest'

    # High intensity + negative state → immediate calming
    if intensity >= 4 and predicted_state in ['overwhelmed', 'restless']:
        return 'box_breathing'

    # Conflicted/mixed + high stress → grounding first
    if predicted_state == 'mixed' and stress_level >= 4:
        return 'grounding'

    # ── STATE-BASED RULES ──
    rules = {
        'calm': {
            'high_energy': 'deep_work',       # energy >= 3
            'low_energy' : 'light_planning'   # energy < 3
        },
        'focused': {
            'high_energy': 'deep_work',
            'low_energy' : 'journaling'
        },
        'neutral': {
            'high_stress': 'grounding',       # stress >= 3
            'low_stress' : 'journaling'
        },
        'restless': {
            'high_stress': 'box_breathing',
            'low_stress' : 'movement'
        },
        'overwhelmed': {
            'high_stress': 'box_breathing',
            'low_stress' : 'rest'
        },
        'mixed': {
            'high_stress': 'grounding',
            'low_stress' : 'journaling'
        }
    }

    state_rules = rules.get(predicted_state, {
        'high_energy': 'journaling',
        'low_energy' : 'rest'
    })

    # Apply energy/stress split
    if predicted_state in ['calm', 'focused']:
        return state_rules['high_energy'] if energy_level >= 3 \
               else state_rules['low_energy']
    else:
        return state_rules['high_stress'] if stress_level >= 3 \
               else state_rules['low_stress']


def decide_when_to_do(time_of_day, predicted_state, intensity):
    """
    Timing logic based on urgency + time of day.
    Urgency = high intensity negative state → act NOW
    """

    urgent   = predicted_state in ['overwhelmed', 'restless'] and intensity >= 4
    positive = predicted_state in ['calm', 'focused']

    if urgent:
        return 'now'

    timing_map = {
        'early_morning': 'within_15_min',
        'morning'      : 'now'       if not positive else 'within_15_min',
        'afternoon'    : 'within_15_min' if not positive else 'later_today',
        'evening'      : 'tonight'   if positive else 'now',
        'night'        : 'now'       if urgent else 'tomorrow_morning'
    }

    return timing_map.get(time_of_day, 'within_15_min')


def generate_message(predicted_state, what_to_do, when_to_do, intensity):
    """
    Template-based supportive message.
    No external API — pure rule-based generation.
    """

    templates = {
        ('overwhelmed', 'box_breathing', 'now'):
            "You seem to be carrying a lot right now. Before anything else, "
            "try 4 slow box breaths — just 2 minutes can ease the weight.",

        ('restless',    'box_breathing', 'now'):
            "There's scattered energy in you today. A quick breathing reset "
            "will help you land before you try to move forward.",

        ('restless',    'movement',      'within_15_min'):
            "That restless energy is actually useful fuel. "
            "A short walk in the next 15 minutes will channel it well.",

        ('calm',        'deep_work',     'now'):
            "You're in a great headspace right now. "
            "This is your window — use it for your most important task.",

        ('calm',        'deep_work',     'within_15_min'):
            "You're settled and clear. Give yourself 10 minutes to set up, "
            "then dive into focused work.",

        ('focused',     'deep_work',     'now'):
            "Your mind feels sharp and ready. "
            "Block the next 90 minutes — deep work will flow naturally.",

        ('neutral',     'journaling',    'within_15_min'):
            "You seem in a balanced place. "
            "A short journal entry might help clarify what you want from today.",

        ('mixed',       'grounding',     'now'):
            "Your signals are a bit mixed right now. "
            "Try a 5-minute grounding exercise before deciding your next step.",
    }

    key = (predicted_state, what_to_do, when_to_do)
    if key in templates:
        return templates[key]

    # Fallback — generic but personalized
    when_readable = when_to_do.replace('_', ' ')
    what_readable = what_to_do.replace('_', ' ')
    return (f"You seem {predicted_state} with intensity {intensity}. "
            f"Try {what_readable} {when_readable} "
            f"to help shift into a better state.")