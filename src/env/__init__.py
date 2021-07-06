from env.cooking import get_cooking_game_env


def get_game_env(game: str, **kwargs):
    assert game in {'cooking'}
    if game == 'cooking':
        return get_cooking_game_env(**kwargs)
