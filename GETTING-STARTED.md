# Getting Started with BotArmy

This guide is for people who want to get BotArmy running but don't have
much (or any) experience with Docker, Kubernetes, or cloud servers. If
you can use the Terminal app and copy-paste a few commands, you can do
this. We'll explain what each step does as we go.

If you've done this kind of thing before, you might prefer the technical
reference: [INSTALL.md](INSTALL.md).

---

## Table of contents

1. [What is BotArmy?](#what-is-botarmy)
2. [Which path is right for you?](#which-path-is-right-for-you)
3. [What you'll need before starting](#what-youll-need-before-starting)
4. [Path A — Run it on your own computer (recommended for first try)](#path-a--run-it-on-your-own-computer)
5. [Path B — Run it on Amazon Web Services (AWS)](#path-b--run-it-on-amazon-web-services-aws)
6. [Path C — Run it on Google Cloud (GCP)](#path-c--run-it-on-google-cloud-gcp)
7. [Talking to BotArmy for the first time](#talking-to-botarmy-for-the-first-time)
8. [When something goes wrong](#when-something-goes-wrong)
9. [Going further](#going-further)

---

## What is BotArmy?

BotArmy is a system of AI agents — software characters with names and
specialities (a Commander, a Researcher, a Coder, a Writer, a
Self-Improver) — that work together on tasks you give them. Under the
hood it uses large language models (Claude, DeepSeek, and others) and
remembers everything across conversations.

To run it, your computer (or a server) needs to start up several pieces:

- **The gateway** — the part that listens for your messages and
  coordinates the agents.
- **A vector database** — stores memories the agents can search.
- **A graph database** — stores relationships between things the agents
  have learned about.
- **A document database** — stores the knowledge corpus.

We use **Docker** to package all of these so you don't have to install
each one separately. Docker is a tool that runs small isolated
"containers" (think mini computers) on your machine. The installer
handles installing Docker for you if you don't have it yet.

---

## Which path is right for you?

| Path | Where it runs | Best for | Cost |
| --- | --- | --- | --- |
| **A. Your computer** | Your laptop or desktop | Trying it out, personal use, learning | Free (uses your computer's electricity) |
| **B. AWS** | Amazon's servers | Always-on access, sharing with a team | ~$210/month |
| **C. GCP** | Google's servers | Same as AWS, slightly cheaper | ~$120/month |

**Recommendation for first-time users: pick Path A.** Get it working on
your computer, play with it, then move to a cloud later if you want
24/7 access. You can always switch — your settings transfer.

You'll need a real Mac or a Linux computer for Path A. Windows users:
install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
(it gives you a Linux environment inside Windows) and follow the Linux
steps. Native Windows is not supported.

---

## What you'll need before starting

### For all paths

You need accounts at these services and a key (a long string of
letters and numbers, like a password) from each. Setting them up takes
about 10 minutes total. The installer will ask you to paste the keys in
when it runs.

| Service | What it does | Cost | Where to get the key |
| --- | --- | --- | --- |
| **Anthropic** | Provides the Claude AI model (the "brain" for important tasks) | Pay per use; ~$5 covers a lot of testing | https://console.anthropic.com/settings/keys |
| **OpenRouter** | Provides cheaper AI models for everyday work | Pay per use; very cheap | https://openrouter.ai/keys |
| **Brave Search** | Lets the agents search the web | Free tier covers 2000 searches/month | https://api.search.brave.com/ |

When you sign up, each site will give you a key starting with `sk-...`,
`sk-or-...`, or `BSA...`. **Save these in a note somewhere safe** — you
won't be able to see them again later.

### What "Terminal" means

You'll see instructions like "open a Terminal and run this". A Terminal
is a window where you type commands instead of clicking. To open it:

- **Mac:** Press `⌘ + Space`, type "Terminal", press Enter.
- **Linux:** Press `Ctrl + Alt + T`, or look for "Terminal" in your apps.
- **Windows (WSL):** Open "Ubuntu" from your Start menu.

When this guide says "run this command", you copy the command, paste it
into the Terminal, and press Enter.

---

## Path A — Run it on your own computer

This puts BotArmy on your Mac or Linux machine. It runs while your
computer is on and stops when you turn it off or close it.

### Step 1 — Get the code

Open Terminal and run:

```bash
git clone https://github.com/nabba/AndrusAI.git
cd AndrusAI
```

If `git` isn't installed, the system will offer to install it. Say yes.

(`git clone` makes a copy of the project on your computer. `cd` moves
into the project folder.)

### Step 2 — Run the installer

```bash
./install.sh
```

That single command does everything:

- Installs **Docker Desktop** (Mac) or **Docker Engine** (Linux) if you
  don't have it. On Mac, the installer will tell you to open Docker
  Desktop manually one time after it installs — do that and re-run
  `./install.sh`.
- Installs Python 3.11+ if needed.
- Asks for your three API keys (paste them in when prompted).
- Generates secure passwords for the databases automatically.
- Downloads about 5 GB of pre-built software packages.
- Starts everything up.

The first run takes **5 to 15 minutes**, depending on your internet
speed. Most of that is downloading. Subsequent runs are seconds.

You'll see a lot of text scrolling by. That's normal. At the end you
should see:

```
✓ BotArmy is up.

  Gateway:   http://127.0.0.1:8765
```

If you see that, **you're done with installation**. Skip to
[Talking to BotArmy for the first time](#talking-to-botarmy-for-the-first-time).

### Step 3 — Check it's working

Run:

```bash
./install.sh --verify
```

You should see green checkmarks next to "postgres", "neo4j", "chromadb",
"gateway HTTP", and "sandbox image". If any are red, see
[When something goes wrong](#when-something-goes-wrong).

### Stopping and restarting

To stop BotArmy:

```bash
docker compose down
```

To start it again:

```bash
docker compose up -d
```

Your data (memories, knowledge) is preserved between restarts. It only
goes away if you run `./install.sh --uninstall` and answer "yes" to the
"wipe data" prompt.

---

## Path B — Run it on Amazon Web Services (AWS)

This deploys BotArmy on Amazon's cloud servers. It runs 24/7 — you can
access it from any computer or phone, and you don't need to keep your
laptop on.

### Before you start

- **AWS account with billing set up.** Sign up at https://aws.amazon.com/.
  AWS will ask for a credit card during signup but the free tier covers
  enough to get started experimenting. The full BotArmy setup costs
  roughly **$210/month** at the cheapest tier.
- **Five command-line tools** installed on your computer: `aws`,
  `terraform`, `kubectl`, `helm`, `docker`. The installer checks for
  them and tells you exactly where to install each from. Easiest on
  Mac: `brew install awscli terraform kubernetes-cli helm`.
- **Your AWS credentials configured** on your computer. Run
  `aws configure` after installing the AWS CLI — it will ask for an
  Access Key ID and Secret Access Key. Generate these in the AWS
  Console under "IAM → Users → Your user → Security credentials".

### Running the install

From the `crewai-team` folder:

```bash
./install.sh --target aws
```

What happens:

1. The installer asks you a few questions (region, cluster name, tier).
   Defaults are sensible — press Enter to accept.
2. It shows you an estimated monthly cost (~$210 cheapest, ~$680 prod).
3. After you confirm, **Terraform** (a tool that creates cloud resources
   from a description file) provisions everything: a network, a
   Kubernetes cluster, a Postgres database, secret storage, a load
   balancer. **This takes 15–25 minutes.** You can leave the Terminal
   window and come back.
4. When that's done, the installer builds the BotArmy gateway image
   and uploads it to AWS.
5. Finally it deploys BotArmy onto the cluster and waits for it to
   become healthy.

When it's finished, you'll see commands you can use to talk to it.

### Tearing it down

When you want to stop paying for it:

```bash
cd deploy/terraform/aws
terraform destroy
```

Type "yes" when prompted. **This wipes the database** — your
memories are gone. Save anything important first by exporting via the
gateway API.

---

## Path C — Run it on Google Cloud (GCP)

Almost identical to AWS, just a different cloud provider. Slightly
cheaper at the small end.

### Before you start

- **GCP account with billing set up.** Sign up at https://cloud.google.com/.
  Google gives new users $300 of free credit that lasts 90 days — more
  than enough to test with.
- **Five command-line tools**: `gcloud`, `terraform`, `kubectl`, `helm`,
  `docker`. On Mac: `brew install --cask google-cloud-sdk` and
  `brew install terraform kubernetes-cli helm`.
- **Your GCP credentials configured.** Run `gcloud auth login` and
  `gcloud auth application-default login` (the second one is what
  Terraform uses).
- **A GCP project ID.** Create one at https://console.cloud.google.com/.
  It's a name like `my-botarmy-12345` — write it down.

### Running the install

```bash
./install.sh --target gcp
```

The installer will ask for your project ID. Everything else works the
same as AWS — answer the questions, confirm the cost (~$120/month
cheapest), wait 15–25 minutes for provisioning, then it deploys
BotArmy automatically.

### Tearing it down

```bash
cd deploy/terraform/gcp
terraform destroy
```

Same warning as AWS — this wipes your databases.

---

## Talking to BotArmy for the first time

Once it's running, open a web browser and go to:

```
http://127.0.0.1:8765
```

(For cloud installs, the installer prints the URL to use instead.)

You should see a simple page confirming the gateway is up. To actually
chat with the agents, you have a few options:

### Option 1: Through Signal (chat app on your phone)

This is how the system was originally designed to be used. Setup takes
about 15 minutes and is documented in the main README under "Signal".
Skip this for now if you just want to try it.

### Option 2: Through the React dashboard

There's a web dashboard at `http://127.0.0.1:3100` (when running
locally). It has a chat box, a memory browser, and a few other panels.

### Option 3: Through the API directly

Useful if you want to test programmatically. Send a POST request to
`/api/cp/tasks` with a JSON body. See the OpenAPI docs at
`http://127.0.0.1:8765/docs`.

**On a laptop install** (Path A) the gateway listens only on `127.0.0.1`,
which is a security boundary on its own — no auth header needed. Just `curl`.

**On a cloud install** (Path B/C) the gateway listens on `0.0.0.0` so
Kubernetes can reach it. To compensate, the installer turns on
**bearer-token auth** for any request that creates or modifies things —
your token is the `GATEWAY_SECRET` value the installer auto-generated. Get
it back later with:

```bash
# AWS
aws secretsmanager get-secret-value --secret-id botarmy-env \
    --query SecretString --output text | jq -r .GATEWAY_SECRET

# GCP
gcloud secrets versions access latest --secret botarmy-env \
    | jq -r .GATEWAY_SECRET

# Either cloud — also stored as a Kubernetes Secret
kubectl -n botarmy get secret botarmy-env \
    -o jsonpath='{.data.GATEWAY_SECRET}' | base64 -d
```

Then send your requests with an `Authorization` header:

```bash
TOKEN="..."   # the value from above
curl -X POST -H "Authorization: Bearer $TOKEN" \
    https://bot.example.com/api/cp/tasks -d '{"text": "research X"}'
```

Read endpoints (GET requests), `/health`, and `/metrics` work without the
header — only mutations need it.

The first task takes 30–60 seconds because the agents need to warm up
and pull memories. Later tasks are faster.

---

## When something goes wrong

### "Permission denied" running `./install.sh`

```bash
chmod +x install.sh
./install.sh
```

(Marks the file as executable. The clone should have done this for you,
but on some systems it doesn't.)

### "command not found: docker"

You need Docker. The installer offers to install it for you on a fresh
run. If it's stuck, install Docker Desktop manually:

- **Mac:** https://www.docker.com/products/docker-desktop/
- **Linux:** https://docs.docker.com/engine/install/

After installing, **open Docker Desktop once** so it can finish setup,
then re-run `./install.sh`.

### Docker is installed but the installer still complains

Docker probably isn't running. On Mac, look for the Docker icon in your
menu bar — it should be steady, not animating. If it's not there, open
the Docker Desktop app from your Applications folder.

On Linux:

```bash
sudo systemctl start docker
```

### "MEM0_POSTGRES_PASSWORD must be set in .env"

The installer should generate this automatically. If you see it, run
`./install.sh` again — it will fill in the missing value.

### Health check times out for "gateway HTTP"

The gateway image is large; first start can take 60–120 seconds even
after the container reports "running". Wait another minute, then run:

```bash
./install.sh --verify
```

Still failing? Check the logs:

```bash
docker compose logs gateway --tail 100
```

Look for lines with `ERROR` or `Exception`. The most common cause is a
typo in your API keys — open `.env` in a text editor and double-check
them. After fixing:

```bash
docker compose up -d gateway
```

### "I want to start completely over"

```bash
./install.sh --uninstall
```

This will ask you twice before wiping data. Then:

```bash
rm .env             # remove your API keys (you'll re-enter them)
./install.sh
```

### Cloud deploy: "postgresql_extension.vector ... could not connect"

This is a known limitation when running Terraform from a corporate
network that blocks outbound database ports. The fix depends on which
cloud:

- **AWS:** See [deploy/terraform/aws/README.md](deploy/terraform/aws/README.md#troubleshooting).
- **GCP:** See [deploy/terraform/gcp/README.md](deploy/terraform/gcp/README.md#troubleshooting).

Both readmes have the workaround: apply infrastructure first, install
the database extension manually from inside the cloud, then re-apply.

### "I'm getting charged more than expected on AWS / GCP"

The cluster's "control plane" bills even when nothing is running on it
(~$73/month on either cloud). To avoid this:

- **Stop using it for a while:** `terraform destroy` in your cloud's
  Terraform folder. This deletes everything except the container
  images. Re-apply when you want it back.
- **Use the local install instead:** Path A doesn't bill anything.

---

## Going further

Once you've got it working:

- **[INSTALL.md](INSTALL.md)** — full technical reference for the
  installer, with all command-line flags and what they do.
- **[deploy/k8s/OBSERVABILITY.md](deploy/k8s/OBSERVABILITY.md)** —
  setting up Grafana dashboards and alert routing (Slack / email /
  Opsgenie). Useful when you've moved to a cloud install and want to
  know if anything goes wrong without checking the logs.
- **[CLAUDE.md](CLAUDE.md)** — high-level architecture: how the agents
  are organised, how the LLM cascade works, what Mem0 does.

If you get stuck somewhere this guide didn't cover, open an issue at
https://github.com/nabba/AndrusAI/issues with:

1. What you were trying to do
2. The exact error message
3. Which path you're on (A, B, or C)
4. The output of `./install.sh --verify` if you're far enough that it works

We'll help.
