from dynaconf import Dynaconf  # type: ignore

settings = Dynaconf(settings_files=[".secrets.toml", "settings.toml"])
