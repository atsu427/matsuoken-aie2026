"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    usage = None
    started_at = time.perf_counter()
    try:
        response = client.models.generate_content(model=MODEL, contents=question)
        usage = response.usage_metadata
    finally:
        response_time = time.perf_counter() - started_at
        if usage is None:
            token_display = "入力・出力・合計トークン数: 未取得"
        else:
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            total_tokens = usage.total_token_count
            input_display = input_tokens if input_tokens is not None else "未取得"
            output_display = output_tokens if output_tokens is not None else "未取得"
            total_display = total_tokens if total_tokens is not None else "未取得"
            token_display = (
                f"入力: {input_display} / 出力: {output_display} / 合計: {total_display} トークン"
            )
        print(
            f"[usage] API呼び出し試行: 1回 / {token_display} "
            f"/ 応答時間: {response_time:.2f} 秒"
        )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
