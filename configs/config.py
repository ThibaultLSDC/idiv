from yaecs import Configuration


class DefaultConfig(Configuration):
    
    @staticmethod
    def get_default_config_path():
        return "./configs/default_config/main_config.yaml"

    def parameters_pre_processing(self):
        return {"*_config_path": self.register_as_additional_config_file}