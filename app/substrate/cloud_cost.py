"""
cloud_cost — pure-Python monthly-spend estimator for GCP/AWS install targets.

Productization plan WP D Phase 0. Used as the operator-visible gate
before any ``terraform apply``. The estimator never calls cloud APIs;
it computes from list prices for the resources the existing Terraform
in ``deploy/terraform/<target>/`` would provision.

Conservative on purpose:
  * Pessimistic = list price, not committed-use, not sustained-use.
  * Operator's actual bill should be ≤ this estimate.
  * Region-specific surcharges captured (europe-north1 is ~10% above
    us-central1 for compute, ~0% for storage).

Cost numbers anchored to the Terraform's own tier comments:
  * ``cheapest`` ≈ $110/mo  (per deploy/terraform/gcp/variables.tf)
  * ``prod``     ≈ $420/mo  (same source)

Refresh cadence: prices drift. The pinning test
(``tests/test_cloud_cost.py``) freezes the tier-total ranges so any
silent rate change in this file is caught — the operator updates the
test only when they have a reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

CloudTarget = Literal["gcp", "aws"]
Tier = Literal["cheapest", "prod"]

# ── Region multipliers ──────────────────────────────────────────────
# Helsinki/Tallinn → europe-north1 (Hamina) is the natural default;
# us-central1 is the reference base.
_GCP_REGION_MULTIPLIER: dict[str, float] = {
    "us-central1":   1.00,
    "us-east1":      1.00,
    "us-west1":      1.00,
    "europe-north1": 1.10,   # ~10% above us-central1
    "europe-west1":  1.13,
    "europe-west3":  1.15,
    "europe-west4":  1.10,
    "asia-northeast1": 1.30,
}

_AWS_REGION_MULTIPLIER: dict[str, float] = {
    "us-east-1":      1.00,
    "us-west-2":      1.00,
    "eu-north-1":     1.05,   # Stockholm — closest to Helsinki
    "eu-central-1":   1.10,
    "eu-west-1":      1.10,
}


# ── Resource line item ──────────────────────────────────────────────


@dataclass(frozen=True)
class LineItem:
    """One billable resource. Cost is monthly USD."""
    category: str          # control_plane | compute | storage | network | monitoring | secrets
    resource: str
    monthly_usd: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CostBreakdown:
    """Itemized estimate for a single (target, tier, region) combination."""
    target: str
    tier: str
    region: str
    enable_monitoring: bool
    has_domain: bool
    line_items: list[LineItem] = field(default_factory=list)

    @property
    def total_monthly_usd(self) -> float:
        return round(sum(li.monthly_usd for li in self.line_items), 2)

    @property
    def total_annual_usd(self) -> float:
        return round(self.total_monthly_usd * 12, 2)

    def by_category(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for li in self.line_items:
            out[li.category] = round(out.get(li.category, 0.0) + li.monthly_usd, 2)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "tier": self.tier,
            "region": self.region,
            "enable_monitoring": self.enable_monitoring,
            "has_domain": self.has_domain,
            "line_items": [li.to_dict() for li in self.line_items],
            "by_category": self.by_category(),
            "total_monthly_usd": self.total_monthly_usd,
            "total_annual_usd": self.total_annual_usd,
        }


# ── Base prices (USD/month, us-central1 list price) ─────────────────


# GKE Autopilot. Zonal control plane is free; regional is $0.10/hr.
_GKE_CONTROL_PLANE_REGIONAL_USD = 73.0     # 0.10 USD/hr * 730 hr

# Autopilot pod-resource pricing. Estimated from the chart's resource
# requests (gateway + postgres + neo4j + chromadb + grafana, optional).
# Conservative — operator's actual requests may be lower.
_AUTOPILOT_VCPU_MONTHLY = 32.5             # 0.0445 USD/vCPU·hr * 730 hr
_AUTOPILOT_MEM_GB_MONTHLY = 3.6            # 0.0049 USD/GB·hr * 730 hr

_CHART_VCPU_REQUEST = {
    "cheapest": 2.0,    # gateway + lightweight memory stack
    "prod":     4.0,    # higher requests, more replicas
}
_CHART_MEM_GB_REQUEST = {
    "cheapest": 4.0,
    "prod":     8.0,
}

# CloudSQL tier list prices (us-central1, monthly).
_CLOUDSQL_TIER_USD = {
    "db-g1-small":        25.0,
    "db-custom-2-7680":  140.0,
}

# CloudSQL disk: PD_SSD $0.17/GB-month list. HA doubles it.
_CLOUDSQL_DISK_GB_MONTHLY = 0.17

# CloudSQL PITR (prod tier) surcharge.
_CLOUDSQL_PITR_SURCHARGE_PCT = 0.25

# Other GCP services (monthly list-price approximations).
_GCP_ARTIFACT_REGISTRY_USD = 1.0          # ~10 GB images at $0.10/GB
_GCP_CLOUD_NAT_USD = 32.0                 # 1 instance + small traffic
_GCP_LOAD_BALANCER_USD = 22.0             # HTTPS LB + 1 forwarding rule
_GCP_MONITORING_USD = 15.0                # kube-prometheus-stack storage + ingestion
_GCP_SECRET_MANAGER_USD = 0.50            # ~8 active secrets

# AWS line-item prices (us-east-1 list, monthly).
_AWS_EKS_CONTROL_PLANE_USD = 73.0         # $0.10/hr × 730 hr
_AWS_NODEGROUP_USD = {
    "cheapest": 90.0,                     # 2× t3.medium spot-ish
    "prod":     220.0,                    # 3× m5.large on-demand
}
_AWS_RDS_USD = {
    "cheapest": 30.0,                     # db.t3.micro
    "prod":     140.0,                    # db.m5.large multi-AZ
}
_AWS_ECR_USD = 1.0
_AWS_NAT_GATEWAY_USD = 36.0
_AWS_LOAD_BALANCER_USD = 22.0


# ── Builders ────────────────────────────────────────────────────────


def _gcp_breakdown(tier: Tier, region: str, enable_monitoring: bool, has_domain: bool) -> list[LineItem]:
    mult = _GCP_REGION_MULTIPLIER.get(region, 1.10)  # default: assume Europe-pricey

    items: list[LineItem] = []

    # 1. GKE control plane
    if tier == "prod":
        items.append(LineItem(
            category="control_plane",
            resource="GKE Autopilot regional control plane",
            monthly_usd=round(_GKE_CONTROL_PLANE_REGIONAL_USD * mult, 2),
            note="regional cluster — multi-zone control plane",
        ))
    else:
        items.append(LineItem(
            category="control_plane",
            resource="GKE Autopilot zonal control plane",
            monthly_usd=0.0,
            note="zonal cluster — control plane free",
        ))

    # 2. Autopilot compute (per-pod requests)
    vcpu = _CHART_VCPU_REQUEST[tier]
    mem_gb = _CHART_MEM_GB_REQUEST[tier]
    compute_usd = round((vcpu * _AUTOPILOT_VCPU_MONTHLY + mem_gb * _AUTOPILOT_MEM_GB_MONTHLY) * mult, 2)
    items.append(LineItem(
        category="compute",
        resource=f"GKE Autopilot pod requests ({vcpu} vCPU + {mem_gb} GB mem)",
        monthly_usd=compute_usd,
        note="billed by sustained pod resource request",
    ))

    # 3. CloudSQL
    sql_tier = "db-custom-2-7680" if tier == "prod" else "db-g1-small"
    sql_base = _CLOUDSQL_TIER_USD[sql_tier]
    if tier == "prod":
        sql_base *= 2.0  # HA = regional, 2× cost
    sql_base *= mult
    disk_gb = 100 if tier == "prod" else 20
    disk_usd = disk_gb * _CLOUDSQL_DISK_GB_MONTHLY
    if tier == "prod":
        disk_usd *= 2.0
    sql_total = round(sql_base + disk_usd, 2)
    if tier == "prod":
        sql_total = round(sql_total * (1 + _CLOUDSQL_PITR_SURCHARGE_PCT), 2)
    items.append(LineItem(
        category="storage",
        resource=f"Cloud SQL {sql_tier} ({disk_gb} GB SSD)" + (" HA + PITR" if tier == "prod" else ""),
        monthly_usd=sql_total,
        note="Postgres + pgvector for Mem0",
    ))

    # 4. Artifact Registry (region-neutral price)
    items.append(LineItem(
        category="storage",
        resource="Artifact Registry (gateway images)",
        monthly_usd=_GCP_ARTIFACT_REGISTRY_USD,
        note="~10 GB image storage",
    ))

    # 5. Cloud NAT
    items.append(LineItem(
        category="network",
        resource="Cloud NAT (egress only)",
        monthly_usd=round(_GCP_CLOUD_NAT_USD * mult, 2),
        note="single NAT instance — egress to providers + Signal-cli",
    ))

    # 6. Load Balancer (only if domain set)
    if has_domain:
        items.append(LineItem(
            category="network",
            resource="HTTPS Load Balancer + managed cert",
            monthly_usd=round(_GCP_LOAD_BALANCER_USD * mult, 2),
            note="public dashboard ingress",
        ))

    # 7. Monitoring (kube-prometheus-stack)
    if enable_monitoring:
        items.append(LineItem(
            category="monitoring",
            resource="kube-prometheus-stack + Grafana",
            monthly_usd=_GCP_MONITORING_USD,
            note="metrics storage + scrape resources",
        ))

    # 8. Secret Manager
    items.append(LineItem(
        category="secrets",
        resource="Secret Manager (8 active secrets)",
        monthly_usd=_GCP_SECRET_MANAGER_USD,
        note="API keys + gateway secret + neo4j + grafana admin",
    ))

    return items


def _aws_breakdown(tier: Tier, region: str, enable_monitoring: bool, has_domain: bool) -> list[LineItem]:
    mult = _AWS_REGION_MULTIPLIER.get(region, 1.10)
    items: list[LineItem] = []

    items.append(LineItem(
        category="control_plane",
        resource="EKS control plane",
        monthly_usd=round(_AWS_EKS_CONTROL_PLANE_USD * mult, 2),
        note="$0.10/hr — unavoidable on EKS",
    ))
    items.append(LineItem(
        category="compute",
        resource=f"EKS node group ({tier})",
        monthly_usd=round(_AWS_NODEGROUP_USD[tier] * mult, 2),
        note="EC2 instances — on-demand list price",
    ))
    items.append(LineItem(
        category="storage",
        resource=f"RDS Postgres ({tier})",
        monthly_usd=round(_AWS_RDS_USD[tier] * mult, 2),
        note="includes 20-100 GB gp3 SSD",
    ))
    items.append(LineItem(
        category="storage",
        resource="ECR (gateway images)",
        monthly_usd=_AWS_ECR_USD,
    ))
    items.append(LineItem(
        category="network",
        resource="NAT Gateway",
        monthly_usd=round(_AWS_NAT_GATEWAY_USD * mult, 2),
        note="hourly + per-GB egress",
    ))
    if has_domain:
        items.append(LineItem(
            category="network",
            resource="ALB + ACM certificate",
            monthly_usd=round(_AWS_LOAD_BALANCER_USD * mult, 2),
        ))
    if enable_monitoring:
        items.append(LineItem(
            category="monitoring",
            resource="kube-prometheus-stack",
            monthly_usd=_GCP_MONITORING_USD,
            note="same chart; storage is on EBS",
        ))
    return items


# ── Public entry ────────────────────────────────────────────────────


def estimate_monthly_cost(
    target: CloudTarget,
    tier: Tier = "cheapest",
    region: str | None = None,
    *,
    enable_monitoring: bool = True,
    has_domain: bool = False,
) -> CostBreakdown:
    """Compute the itemized monthly cost for a cloud install.

    Args:
      target: cloud target (``gcp`` or ``aws``).
      tier: ``cheapest`` or ``prod`` (matches Terraform's ``var.tier``).
      region: cloud region. None resolves to the Helsinki-closest
        default for the chosen target.
      enable_monitoring: whether kube-prometheus-stack is installed.
      has_domain: whether a public ingress + cert is provisioned.

    Returns:
      ``CostBreakdown`` with per-line itemization. All numbers in USD/month,
      list price, no committed-use discount.
    """
    if target not in ("gcp", "aws"):
        raise ValueError(f"unknown cloud target: {target!r}")
    if tier not in ("cheapest", "prod"):
        raise ValueError(f"unknown tier: {tier!r}")

    if region is None:
        region = "europe-north1" if target == "gcp" else "eu-north-1"

    if target == "gcp":
        items = _gcp_breakdown(tier, region, enable_monitoring, has_domain)
    else:
        items = _aws_breakdown(tier, region, enable_monitoring, has_domain)

    return CostBreakdown(
        target=target,
        tier=tier,
        region=region,
        enable_monitoring=enable_monitoring,
        has_domain=has_domain,
        line_items=items,
    )


def format_breakdown(b: CostBreakdown) -> str:
    """Human-readable summary suitable for the CLI."""
    lines: list[str] = []
    lines.append(f"=== Estimated monthly cost — {b.target} / tier={b.tier} / region={b.region} ===")
    lines.append("")
    cat_max = max(len(li.category) for li in b.line_items) if b.line_items else 0
    for li in b.line_items:
        note = f"  ({li.note})" if li.note else ""
        lines.append(f"  {li.category:<{cat_max}}  ${li.monthly_usd:>7.2f}  {li.resource}{note}")
    lines.append("")
    lines.append("  ─" * 8)
    lines.append(f"  TOTAL                          ${b.total_monthly_usd:>7.2f} / month")
    lines.append(f"                                 ${b.total_annual_usd:>7.2f} / year")
    lines.append("")
    lines.append("  Notes:")
    lines.append("   • Conservative — list price, no committed-use discount.")
    lines.append("   • Excludes data egress beyond NAT defaults.")
    lines.append("   • Excludes LLM provider spend (Anthropic, OpenRouter, etc.).")
    return "\n".join(lines)
