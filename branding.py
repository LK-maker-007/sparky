"""Branding utilities for Sparky."""

from __future__ import annotations

from typing import Optional

import config


def build_branded_blocks(
    message: str, *, extra_blocks: Optional[list[dict]] = None
) -> Optional[list[dict]]:
    """Return Slack blocks that include the Sparky logo (if configured)."""

    blocks: list[dict] = []
    logo_url = config.LOGO_URL
    if logo_url:
        blocks.append(
            {
                "type": "image",
                "image_url": logo_url,
                "alt_text": "Sparky logo",
            }
        )

    if extra_blocks:
        blocks.extend(extra_blocks)
    else:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": message}})

    return blocks or None
