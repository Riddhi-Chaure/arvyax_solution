def decide_what_to_do(predicted_state, intensity, stress_level, energy_level):

    if energy_level <= 1:
        return 'rest'
    if intensity >= 4 and predicted_state in ['overwhelmed', 'restless']:
        return 'box_breathing'
    if predicted_state == 'mixed' and stress_level >= 4:
        return 'grounding'

    rules = {
        'calm':('deep_work','light_planning'),# (high_energy, low_energy)
        'focused':('deep_work','journaling'),
        'neutral':('grounding','journaling'),# (high_stress, low_stress)
        'restless':('box_breathing','movement'),
        'overwhelmed':('box_breathing','rest'),
        'mixed':('grounding','journaling'),
    }

    hi,lo = rules.get(predicted_state,('journaling', 'rest'))

    if predicted_state in ['calm','focused']:
        return hi if energy_level >= 3 else lo
    return hi if stress_level >= 3 else lo


def decide_when_to_do(time_of_day, predicted_state, intensity):

    if predicted_state in ['overwhelmed','restless'] and intensity >= 4:
        return'now'

    positive = predicted_state in ['calm','focused']

    when = {
        'early_morning':'within_15_min',
        'morning':'within_15_min' if positive else 'now',
        'afternoon':'later_today'  if positive else 'within_15_min',
        'evening':'tonight' if positive else 'now',
        'night':'tomorrow_morning',
    }

    return when.get(time_of_day,'within_15_min')


def generate_message(predicted_state, what_to_do, when_to_do, intensity):

    templates = {
        ('overwhelmed', 'box_breathing', 'now'):
            "You're carrying a lot right now. Try 4 slow box breaths first "
            "just 2 minutes can take the edge off.",

        ('restless', 'box_breathing', 'now'):
            "There's scattered energy in you today. A quick breathing reset "
            "will help you land before moving forward.",

        ('restless', 'movement', 'within_15_min'):
            "That restless energy is actually useful "
            "a short walk in the next 15 minutes will channel it.",

        ('calm', 'deep_work', 'now'):
            "You're in a good headspace. This is your window " 
            "use it for your most important task.",

        ('calm', 'deep_work', 'within_15_min'):
            "You're settled. Give yourself 10 minutes to set up "
            "then get into focused work.",

        ('focused', 'deep_work', 'now'):
            "Mind feels sharp. Block the next 90 minutes "
            "deep work will flow.",

        ('neutral', 'journaling', 'within_15_min'):
            "You're in a balanced place. A short journal entry "
            "might help clarify what you want from today.",

        ('mixed', 'grounding', 'now'):
            "Signals are a bit mixed. Try 5 minutes of grounding "
            "before deciding your next step.",
    }

    msg = templates.get((predicted_state, what_to_do, when_to_do))
    if msg:
        return msg

    return (f"You seem {predicted_state} right now (intensity {intensity}). "
            f"Try {what_to_do.replace('_',' ')} "
            f"{when_to_do.replace('_',' ')}.")