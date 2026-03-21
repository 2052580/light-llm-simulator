import argparse
from conf.config import Config
from src.search.deepep import DeepEpSearch


def add_default_mode_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--serving_mode', type=str, default="DeepEP")
    parser.add_argument('--model_type', type=str, default="Qwen/Qwen3.5-397B-A17B")
    parser.add_argument('--device_type', type=str, default="Ascend_A3Pod")
    parser.add_argument('--min_attn_bs', type=int, default=2)
    parser.add_argument('--max_attn_bs', type=int, default=1000)
    parser.add_argument('--min_die', type=int, default=16)
    parser.add_argument('--max_die', type=int, default=768)
    parser.add_argument('--die_step', type=int, default=16)
    parser.add_argument('--tpot', nargs='+', type=int, default=[50])
    parser.add_argument('--kv_len', nargs='+', type=int, default=[4096])
    parser.add_argument('--micro_batch_num', nargs='+', type=int, default=[3])
    parser.add_argument('--next_n', type=int, default=1)
    parser.add_argument('--multi_token_ratio', type=float, default=0.7)
    parser.add_argument('--attn_tensor_parallel', type=int, default=1)
    parser.add_argument('--ffn_tensor_parallel', type=int, default=1)


def run_search(args):
    for mbn in args.micro_batch_num:
        for tpot in args.tpot:
            for kv_len in args.kv_len:
                config = Config(
                    serving_mode=args.serving_mode,
                    model_type=args.model_type,
                    device_type=args.device_type,
                    min_attn_bs=args.min_attn_bs,
                    max_attn_bs=args.max_attn_bs,
                    min_die=args.min_die,
                    max_die=args.max_die,
                    die_step=args.die_step,
                    tpot=tpot,
                    kv_len=kv_len,
                    micro_batch_num=mbn,
                    next_n=args.next_n,
                    multi_token_ratio=args.multi_token_ratio,
                    attn_tensor_parallel=args.attn_tensor_parallel,
                    ffn_tensor_parallel=args.ffn_tensor_parallel
                )
                DeepEpSearch(config).deployment()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_default_mode_arguments(parser)
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
