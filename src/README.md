## Directory Breakdown
```js
├── agent.py // Agent file, initials policy or target models, memory, belief graph model, LTL updater, computes loss, etc.
├── args.py // arguments file for command line interface
├── belief_graph.py // belief graph data class (not the belief graph updater)
├── components.py // various support modules
├── env // all code relating to environments
│   ├── __init__.py
│   └── cooking.py // the Cooking domain environment
├── evaluate.py
├── experience_replay.py
├── graph_updater.py // this is the belief graph update model code
├── logic.py // various logic data classes
├── ltl // all code relating to LTL
│   ├── __init__.py // holds the main LTL dataclass
│   ├── progression.py // performs LTL progression
│   └── translator.py // translates observations to LTL
├── main.py // main command line interface entrypoint
├── model // the Policy model code
│   ├── features.py // holds features for TextEncoder
│   ├── __init__.py // the main model is defined here + action selector
│   ├── layers.py // various support layers for models
│   ├── pretrained_lm.py // returns pretrained LM from Transformers library
│   └── utils.py // various support utils for models
├── optim // returns the Optimizers for learning
│   ├── __init__.py
│   └── radam.py
├── segment_tree.py // Segment tree for experience replay
├── state.py // State dataclass for storing in memory
├── test.py // Testing file
├── train.py // Training file
└── utils.py // various utils that support the training or testing pipline
```

### GATA Models
Our code also relies on the GATA pre-trained belief graph updaters. We refer the user to [GATA-repo](https://github.com/xingdi-eric-yuan/GATA-public) to see how to train those models from scratch. 

We do however provide pre-trained weights for the GATA models that we used. They can be found in [support-files/gata-models](support-files/gata-models).

We also provide the script we used to augment the dataset for training the GATA models. The script can be found at [support-files/gata-models/augment_dataset.py](support-files/gata-models/augment_dataset.py). It should be run in the top-level directory of the [GATA code.](https://github.com/xingdi-eric-yuan/GATA-public).

## Training and Testing
For training and testing, run the following in the command line:
`python main.py --train --test --config *path-to-config.yaml*`

For only training, run the following in the command line:
`python main.py --train --config *path-to-config.yaml*`

For only testing, run the following in the command line:
`python main.py --test --config *path-to-config.yaml*`

We also provide the ability to overrride settings in the config file using the `--params` or `-p` argument. 

This argument follows the following format:
```console
  -p my.setting=value [my.setting=value ...], --params my.setting=value [my.setting=value ...]
                        override params of the config file, e.g. -p 'training.batch_size=100'
```

For example, in the below description of the config file.
```yaml
io:
  ...
  root: 'root-dir'
  ...
```
can be overwritten as follows:
```console
python main.py --train --test --config *path-to-config.yaml* -p io.root="new-root-dir-name"
```

## Where to find config files
We provide a comprehensive set of config files to run all experiments presented in the paper in [support-files/paper-configs](support-files/paper-configs). Note that paths in the config files can be updated, depending on where you would like to save outputs and where you download the various support files (dataset, embeddings, etc.).

Specifically, the files for each experiment are as follows:
- For the general performance using the 100 and 20 game training sets, see [support-files/paper-configs/20-games](support-files/paper-configs/20-games) and [support-files/paper-configs/100-games](support-files/paper-configs/100-games)
- For the config files to run GATA-GTP with stripped instructions, see [support-files/paper-configs/stripped-instructions](support-files/paper-configs/stripped-instructions)
- For the config files to run GATA-GTP with forced examination of the cookbook, see [support-files/paper-configs/examining-the-cookbook](support-files/paper-configs/examining-the-cookbook)
- For the config files to run LTL-GATA without the LTL reward only, without LTL-based termination only, and without both the LTL reward and LTL-based termination, see [support-files/paper-configs/ltl-reward](support-files/paper-configs/ltl-reward), [support-files/paper-configs/ltl-termination](support-files/paper-configs/ltl-termination), and [support-files/paper-configs/ltl-reward-termination](support-files/paper-configs/ltl-termination-reward), respectively
- For the config files to run LTL-GATA without progression, see [support-files/paper-configs/ltl-progression](support-files/paper-configs/ltl-progression)
- For the config files to run LTL-GATA with the multi-token predicate format, see see [support-files/paper-configs/ltl-format](support-files/paper-configs/ltl-format)

## Understanding the config.yaml file
The first block of configs deal with input/output parameters:
```yaml
io:
  tag: null # if null, tag will be set to dir name of root (below)
  root: 'root-dir' # root directory to store output files
  output_dir: 'output' # name of output files (results.csv, test_results.csv)
  checkpoint_dir: 'checkpoints' # name of checkpoints dir to save to
  trajectories_dir: 'trajectories' # name of trajectories dir to save to
  data_dir: "???/rl.0.2" # path to game files
  vocab_dir: "???/vocabularies" # path to vocab files
  report_frequency: 1000  # episode
  report_history_length: 500 # length of cache you report a running avg. over
  save_trajectories_frequency: -1 # frequency of saving trajectories
  pretrained_embedding_path: "???/crawl-300d-2M.vec.h5" # path to word embeddings
```

This next block has pre-training configs.
They are unused in our code as we did not implement any pre-training.
```yaml
pretrain:
  ltl: true
  batch_size: 128
  max_episodes: 10000
  steps_per_episode: 50
  learn_from_this_episode: -1
```

This block deals with all testing configs:
```yaml
test:
  filename: 'checkpoints/best_eval.pt' # name of checkpoint file to run. Not this file will be prepended by the root directory.
  game: 'cooking' # name of the game (we only use 'cooking')
  batch_size: 20 # number of games to play in parallel (20 since 20 testing games)
  difficulty_level: 3
  steps_per_episode: 100
  feed_cookbook_observation: False # whether to force first action in game to be 'examine cookbook'
  softmax: False # toggle for boltzmann action selection
  softmax_temperature: 100
  eps: 0.0 # whether to use eps-greedy in testing action selection
```

This block defines config for evaluation on the validation set during training:
```yaml
evaluate:
  run: True # whether to run or not
  game: 'cooking'
  batch_size: 20 # game to parallelize (only 20 validation games)
  frequency: 1000 # how often during training (in episodes)
  adjust_rewards: False
  difficulty_level: 3
  steps_per_episode: 100
  feed_cookbook_observation: False # whether to force first action in game to be 'examine cookbook'
  eps: 0 # whether to use eps-greedy in testing action selection
```

This block defines config for checkpointing during training:
```yaml
checkpoint:
  load: False # whether you should load the checkpoint at the start of training
  resume: False # whether you should resume from that checkpoint (i.e. the same episode no)
  filename: 'best.pt' # the name of the checkpoint (to load from, if you toggle to do so)
  save_frequency: 1000  # how often (in episodes) to save
  save_each: False # whether to save a unique checkpoint file or simply overwrite the 'latest.pt' file
```

This block defines config for all things training:
```yaml
training:
  game: 'cooking'
  cuda: True
  mlm_alpha: 1 # can ignore - experimental
  batch_size: 50 # parallelize of gameplay batch size
  random_seed: 123
  max_episode: 100000
  light_env_infos: False # env infos from TextWorld - light = faster training
  penalize_path_length: -1 # -1 to disable - unsused in our experiments
  use_negative_reward: False # whether to add neg. reward for failing a game - unused
  persistent_negative_reward: False # whether to persist a neg reward from LTL - experimental - unused
  end_on_ltl_violation: True # end episode of LTL is violated (LTL-based termination)
  reward_ltl: True # LTL-based reward bonus
  reward_ltl_only: False # whether to only reward from LTL bonus (and not TextWorld) - experimental - unused
  reward_ltl_positive_only: False # Only include positive LTL reward bonus - experimental - unused
  backwards_ltl: false # experimental - unused
  reward_per_ltl_progression: False # whether to reward per stage in LTL progression - unused
  optimizer: # optim config...
    name: 'adam'
    kwargs:
      lr: 0.0003
    scheduler: null
    scheduler_kwargs:
      gamma: 0.95
      step_size: 30000
    clip_grad_norm: 5
    learning_rate_warmup:  0
    fix_parameters_keywords: []
  patience: 3  # >=1 to enable - reloads from best.pt if validatino scores decrease 3 times in a row
  difficulty_level: 3  # 1=0, 3=1, 7=2, 5=3 (mapping of games from TextWorld to games in paper)
  training_size: 20  # 20, 100 - size of training game set
  game_limit: -1 # whether to limit games set size (-1 to toggle off) - unused
  steps_per_episode: 50
  update_per_k_game_steps: 1 # because we play batch_size = 50 in parallel, this is like updating every 50 game steps
  learn_from_this_episode: 1000 # training warmup period
  target_net_update_frequency: 500 # episode frequency for updating target net
  context_length: 1 # context length of observations
  graph_reward_lambda: -1 # experimental - unused
  graph_reward_filtered: False # experimental - unused
  randomized_nouns_verbs: False # experimental - unused
  all_games: False # experimental - unused
  feed_cookbook_observation: False # whether to force first action in game to be 'examine cookbook'
  strip_instructions: False # whether to strip instructinos from observations (for ablation study)
  prune_actions: False # experimental - unsused

  experience_replay: # experience replay config...
    eps: 0.000001
    beta: 0.4
    alpha: 0.6
    multi_step: 1
    batch_size: 64
    capacity: 500000
    buffer_reward_threshold: 0.1
    sample_update_from: 4
    discount_gamma_game_reward: 0.9
    sample_history_length: 8 # for recurrent
    accumulate_reward_from_final: False

  epsilon_greedy: # eps greedy config
    anneal_from: 1.0
    anneal_to: 0.1
    episodes: 3000  # -1 if not annealing
```

This block defines config for LTL updater:
```yaml
ltl:
  use_ground_truth: False # ground truth only used in idea testing - should always be False
  reward_scale: 1 # scale for reward bonus (always 1)
  as_bonus: True # for reward
  next_constrained: False # whether to constrain LTL using next operator - always False
  single_token_prop: True # togle single or multi token prop - for ablation 
  incomplete_cookbook: False # unused
  no_cookbook: False # whether to remove the cookbook as an LTL instructino - experimental - unused
  single_reward: False # whether to only give a single reward once all instructions are done
  negative_for_fail: False # unused
  dont_progress: False # whether not to progress the instructions
```

The next block defines the general model config.
It can toggle whether to only use observations (TDQN), observations + graph (GATA), or observations + graph + LTL (LTL-GATA).
```yaml
model:
  use_ltl: False # toggle LTL usage
  inverse_dynamics_loss: False # experimental - unused
  use_independent_actions_encoder: True # whether to have a seperate actions encoder - always True
  recurrent_memory: False # whether to use recurrent model - false in our experiments always
  use_observations: True # whether to use observations
  use_belief_graph: True # whether to use belief graph (GATA)
  concat_features: True # whether to concat the features
  use_pretrained_lm_for_actions: False # experimental - unused
  concat_strings: False # experimental - unused
  same_ltl_text_encoder: False # whether to use same encoder for observations + LTL - always false
  action_net_hidden_size:  64 # latent dimension D
```

The next series of blocks defines the actual model parameters themselves, as discussed in the supplementary material.
```yaml
actions_encoder:
  lstm_backbone: False # experimental - unused
  mlp_backbone: False # experimental - unused
  one_hot_encoding: False # experimental - unused
  use_pretrained_lm: False # experimental - unused
  pretrained_lm_name: 'bert'
  pretrained_lm_checkpoint: null
  trainable: True
  self_attention: True
  position_encoding: "cossine"
  mlm_loss: False
  ### Details
  word_embedding_size: 300
  encoder_conv_num: 5
  num_encoders: 1
  n_heads: 1
  action_net_hidden_size:  64
  pretrained_embedding_path: "???/crawl-300d-2M.vec.h5" # path to word embeddings
```

```yaml
text_encoder:
  lstm_backbone: False # experimental - unused
  mlp_backbone: False # experimental - unused
  one_hot_encoding: False # experimental - unused
  use_pretrained_lm: False # experimental - unused
  pretrained_lm_name: 'bert'
  pretrained_lm_checkpoint: null
  trainable: True
  self_attention: True
  position_encoding: "cossine"
  mlm_loss: False # experimental - unused
  ### Details
  word_embedding_size: 300
  encoder_conv_num: 5
  num_encoders: 1
  n_heads: 1
  action_net_hidden_size:  64
  pretrained_embedding_path: "???/crawl-300d-2M.vec.h5" # path to word embeddings
```

```yaml
ltl_encoder:
  lstm_backbone: False # experimental - unused
  mlp_backbone: False # experimental - unused
  one_hot_encoding: False # experimental - unused
  use_pretrained_lm: False # experimental - unused
  pretrained_lm_name: 'bert'
  pretrained_lm_checkpoint: null
  trainable: True
  self_attention: True
  position_encoding: "cossine"
  mlm_loss: False # experimental - unused
  ### Details
  word_embedding_size: 300
  encoder_conv_num: 5
  num_encoders: 1
  n_heads: 1
  action_net_hidden_size:  64
  pretrained_embedding_path: "???/crawl-300d-2M.vec.h5" # path to word embeddings
```

```yaml
graph_updater:
  # checkpoint: '???/gata_pretrain_obs_infomax_model.pt' # ckpt to GATA COC model
  checkpoint: '???/gata_pretrain_cmd_generation_model.pt' # ckpt to GATA GTP model
  from_pretrained: True
  use_self_attention: False
  max_target_length: 200
  n_heads: 1
  real_valued: False # True if using GATA COC, False if using GATA GTP
  block_dropout: 0.
  encoder_layers: 1
  decoder_layers: 1
  encoder_conv_num: 5
  block_hidden_dim: 64
  attention_dropout: 0.
  embedding_dropout: 0.
  node_embedding_size: 100
  relation_embedding_size: 32
  gcn_dropout: 0.
  gcn_num_bases: 3
  gcn_highway_connections: True
  gcn_hidden_dims: [64, 64, 64, 64, 64, 64]  # last one should equal to block hidden dim
  use_ground_truth_graph: False
  word_embedding_size: 300
  embedding_dropout: 0.
```
