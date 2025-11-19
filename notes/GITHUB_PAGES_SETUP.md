# Setting up GitHub Pages for Documentation

This repository is configured to automatically build and publish Sphinx documentation to GitHub Pages.

## Automatic Deployment

The documentation is automatically built and deployed when:
- Code is pushed to the `master` branch
- A pull request is merged to `master`
- The workflow is manually triggered

## GitHub Repository Settings

To enable GitHub Pages, configure the following in your repository:

1. Go to **Settings** → **Pages**
2. Under **Build and deployment**:
   - **Source**: Select "GitHub Actions"
3. The documentation will be available at: `https://jacobwilliams.github.io/vibehdf5/`

## Workflow Details

The workflow (`.github/workflows/docs.yml`) performs these steps:

1. **Build Job**:
   - Checks out the repository
   - Sets up Python 3.11
   - Installs package with documentation dependencies
   - Builds HTML documentation with Sphinx
   - Uploads the built docs as an artifact

2. **Deploy Job** (only on master branch):
   - Takes the built documentation
   - Deploys it to GitHub Pages

## Manual Trigger

You can manually trigger a documentation rebuild:

1. Go to **Actions** tab in your repository
2. Select "Build and Deploy Documentation"
3. Click "Run workflow"
4. Choose the branch and click "Run workflow"

## Local Testing

Before pushing, test documentation locally:

```bash
cd docs
make html
open _build/html/index.html
```

## Troubleshooting

If the deployment fails:

1. Check the Actions tab for error messages
2. Ensure GitHub Pages is enabled in repository settings
3. Verify that the workflow has proper permissions
4. Check that all dependencies are correctly specified in `pyproject.toml`

## First Time Setup

After enabling GitHub Pages:

1. Push changes to trigger the workflow
2. Wait for the action to complete (check Actions tab)
3. Visit `https://jacobwilliams.github.io/vibehdf5/` to see your docs

The first deployment may take a few minutes to propagate.
