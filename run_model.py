from unityagents import UnityEnvironment

from continuous_control import Environment, plot_scores


def run_model():
    unity_env_file = 'Reacher_Linux_20_agents/Reacher.x86_64'
    env = Environment(UnityEnvironment(file_name=unity_env_file))
    scores = env.run_model('actor.pth')
    env.close()


if __name__ == "__main__":
    run_model()
