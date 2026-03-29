from app.core.feature_extractor import RequestFeatures
from app.schemas.router_policy import PolicyConfig

RouteClass = str  # "local_fast" | "cloud_fast" | "cloud_debug" | "cloud_architecture" | "cloud_vision"


def route(features: RequestFeatures, policy: PolicyConfig) -> RouteClass:
    rules = policy.routing
    kw = rules.keyword_rules

    # Rule 1: image → cloud_vision
    if features.has_image:
        return "cloud_vision"

    # Rule 2: vision keywords
    if any(k in features.all_text for k in kw.get("cloud_vision", [])):
        return "cloud_vision"

    # Rule 3: architecture keywords or very long prompt
    if (
        any(k in features.all_text for k in kw.get("cloud_architecture", []))
        or features.total_chars >= rules.thresholds.cloud_architecture_min_chars
    ):
        return "cloud_architecture"

    # Rule 4: debug keywords
    if any(k in features.all_text for k in kw.get("cloud_debug", [])):
        return "cloud_debug"

    # Rule 5: local_fast keywords + short context
    if (
        any(k in features.all_text for k in kw.get("local_fast", []))
        and features.total_chars <= rules.thresholds.local_fast_max_chars
        and features.message_count <= rules.thresholds.local_fast_max_messages
    ):
        return "local_fast"

    # Rule 6: short prompt + few messages → local_fast
    if (
        features.total_chars <= rules.thresholds.local_fast_max_chars
        and features.message_count <= rules.thresholds.local_fast_max_messages
    ):
        return "local_fast"

    # Default
    return rules.default_route
