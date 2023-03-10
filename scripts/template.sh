python main.py \ 
    # env id 
    --env_id ant_dir_varibad \

    # cluster
    --num_prototypes 10 \
    --lr_cluster 0.00003 \
    --num_cluster_updates 3 \
    --cluster_anneal_lr False \
    --cluster_batch_num_trajs 1000 \

    # ablation 
    --disable_metalearner False \
    --disable_kl_term False \
    --disable_cluster True \