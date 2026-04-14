import yaml
import os


class everyThingOptions():
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.opt = None

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        self.opt = self.DictToObj(config_dict)
        
        # 执行打印与保存
        self.print_and_save_options(config_dict)
        return self.opt

    class DictToObj:
        def __init__(self, entries):
            for k, v in entries.items():
                if isinstance(v, dict):
                    self.__dict__[k] = everyThingOptions.DictToObj(v)
                else:
                    self.__dict__[k] = v

    def print_and_save_options(self, opt_dict):
        message = '\n'
        message += '----------------- Loaded Config -----------------\n'
        
        # 递归遍历字典并格式化打印
        def format_msg(d, indent=0):
            msg = ""
            for k, v in d.items():
                if isinstance(v, dict):
                    msg += ' ' * indent + f"{k}:\n"
                    msg += format_msg(v, indent + 4)
                else:
                    msg += ' ' * indent + '{:<20}: {:<30}\n'.format(str(k), str(v))
            return msg

        message += format_msg(opt_dict)
        message += '----------------------- End ---------------------\n'
        print(message)

        save_dir = os.path.join(opt_dict.get('checkpoints_dir', './checkpoints'), opt_dict.get('name', 'default'))
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'opt_log.txt'), 'w', encoding='utf-8') as f:
            f.write(message)
    
if __name__ == "__main__":
    base_options: everyThingOptions = everyThingOptions("D:/projects/digitalFilm/options/digitalFilm.yaml")
    base_options.load_config()
    