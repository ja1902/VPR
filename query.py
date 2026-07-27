"""
Unified CLI for Visual Place Recognition queries.

Usage:
    python query.py image <photo.jpg> [--rerank] [--top_k 10]
    python query.py video <clip.mp4> [--rerank] [--fps 2] [--max_frames 15] [--crop 10,25]
"""
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Visual Place Recognition — geolocate a photo or video",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── image ────────────────────────────────────────────────────
    img_parser = sub.add_parser("image", help="Query with a single photo")
    img_parser.add_argument("path", help="Path to the query image")
    img_parser.add_argument("--top_k", type=int, default=10)
    img_parser.add_argument("--rerank", action="store_true",
                            help="Adaptive Pair-VPR re-ranking")

    # ── video ────────────────────────────────────────────────────
    vid_parser = sub.add_parser("video", help="Query with a video clip")
    vid_parser.add_argument("path", help="Path to the query video")
    vid_parser.add_argument("--top_k", type=int, default=10)
    vid_parser.add_argument("--rerank", action="store_true",
                            help="Adaptive Pair-VPR re-ranking")
    vid_parser.add_argument("--fps", type=float, default=2.0,
                            help="Frame sampling rate (default: 2.0)")
    vid_parser.add_argument("--max_frames", type=int, default=15,
                            help="Max frames to extract (default: 15)")
    vid_parser.add_argument("--crop", type=str, default=None,
                            help="Crop top%%,bottom%% for dashcam HUD (e.g. --crop 10,25)")

    args = parser.parse_args()

    if args.mode == "image":
        from vpr.query_image import query
        query(args.path, args.top_k, args.rerank)

    elif args.mode == "video":
        crop_pct = None
        if args.crop:
            parts = args.crop.split(",")
            crop_pct = (float(parts[0]), float(parts[1]))

        from vpr.query_video import query_video
        query_video(
            args.path, args.fps, args.max_frames, args.top_k, args.rerank, crop_pct,
        )


if __name__ == "__main__":
    main()
