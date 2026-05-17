# Cosign attestor pipeline (Binary Authorization)

When `hardening_profile=strict`, the GKE cluster runs Binary Authorization.
Default mode `AUDIT` logs would-be-blocks but lets unsigned images through.
Before flipping to `ENFORCE` you must wire the cosign attestor — otherwise
the next deploy will fail at pod admission with `denied by Binary Authorization`.

## One-time setup

```bash
brew install cosign     # or download from sigstore.dev/cosign
./scripts/install/cosign_setup.sh \
    --project-id botarmy-495107 \
    --confirm "SET UP COSIGN"
```

What this does (idempotent — safe to re-run):

1. Generates a cosign keypair under `deploy/k8s/binauthz/`
   (`cosign.key` mode 600, `cosign.pub` mode 644).
2. Enables `containeranalysis.googleapis.com` + `binaryauthorization.googleapis.com`.
3. Creates a Container Analysis NOTE `botarmy-attestor-note`.
4. Creates the Binary Authorization ATTESTOR `botarmy-attestor` bound to that note.
5. Uploads `cosign.pub` to the attestor as a PKIX P-256 SHA-256 key.

After this:
* `cosign.key` lives ONLY on your workstation — never commit it.
  The `.gitignore` in `deploy/k8s/binauthz/` excludes it by default.
* The attestor exists in GCP and is empty (no images attested yet).
* You can already point terraform at it: in
  `workspace/migrations/<run>/terraform.tfvars` set
  `binauthz_attestor_name = "botarmy-attestor"`.

## Signing each image

After `docker push`, capture the resulting digest and call:

```bash
COSIGN_PASSWORD='your-cosign-password' \
botarmy hardening sign-image \
    europe-north1-docker.pkg.dev/botarmy-495107/botarmy-gateway/gateway@sha256:<digest> \
    --project-id botarmy-495107
```

This runs two steps:
1. `cosign sign --key cosign.key <image>` — adds the OCI signature layer.
2. `gcloud container binauthz attestations create` — records the
   attestation against `botarmy-attestor`.

Both succeed → the image is admissible under ENFORCE.

## Promoting to ENFORCE

Once you've verified at least one signed image:

1. Re-run terraform with `binauthz_attestor_name = "botarmy-attestor"`
   so the policy resource references the attestor.
2. In React `/cp/settings` → Cloud hardening → Binary Authorization mode
   → click `ENFORCE` and type `ENFORCE BINAUTHZ` to confirm.
3. Re-run terraform; the policy flips to `ENFORCED_BLOCK_AND_AUDIT_LOG`
   with `REQUIRE_ATTESTATION` for non-allowlisted images.

If the next pod admission fails, your CI/CD signing isn't reliable yet —
flip the mode back to AUDIT in Settings.

## CI/CD integration

In your build pipeline, after `docker push`:

```yaml
# .github/workflows/release.yml fragment
- name: cosign sign + attest
  env:
    COSIGN_PASSWORD: ${{ secrets.COSIGN_PASSWORD }}
  run: |
    IMAGE="${{ env.IMAGE_REPO }}/gateway@${{ steps.push.outputs.digest }}"
    cosign sign --key cosign.key --yes "$IMAGE"
    gcloud container binauthz attestations create \
      --artifact-url "$IMAGE" \
      --attestor botarmy-attestor \
      --attestor-project "${{ env.GCP_PROJECT }}" \
      --signature-file <(cosign sign --key cosign.key --output-signature - "$IMAGE") \
      --public-key-id //cosign.pub:botarmy-attestor
```

Store `cosign.key` + `COSIGN_PASSWORD` as GitHub secrets.

## What about AWS?

AWS EKS doesn't ship an image-signing admission webhook. Equivalent
options (out of scope for the initial slice):

* **Kyverno** — third-party policy controller; verifies cosign
  signatures at admission. Install via Helm.
* **AWS Signer** — for Lambda. Doesn't help EKS.
* **OCI signature verification at pull** — Notation + ECR Repository
  policy can verify on pull, not on admission.

For now, AWS clusters rely on:
* ECR `image_tag_mutability = IMMUTABLE` (set by terraform when
  `hardening_profile=strict`) — once a tag is pushed it can't be replaced.
* ECR scan-on-push (already on) — catches known CVEs.
* WAFv2 + Cloud Armor-style protections at the ingress layer.

## Operator checklist

- [ ] `./scripts/install/cosign_setup.sh --project-id <proj> --confirm "SET UP COSIGN"`
- [ ] Verify `deploy/k8s/binauthz/cosign.key` is mode 600 + gitignored
- [ ] Add `binauthz_attestor_name = "botarmy-attestor"` to the next
      `terraform.tfvars` (or set via `/cp/settings`)
- [ ] Sign your first image: `botarmy hardening sign-image ...`
- [ ] Verify in GCP console → Binary Authorization → Attestations
- [ ] Flip Settings → ENFORCE
- [ ] Re-deploy and confirm pods admit; if not, flip back to AUDIT
