from dynaconf import Dynaconf  # type: ignore

settings = Dynaconf(
    settings_files=["../config/.secrets.toml", "../config/settings.toml"]
)
