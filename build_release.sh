#!/usr/bin/env bash
set -e
echo "Building frontend..."
cd web/frontend && npm run build
cd ../..
echo "Copying static assets..."
rm -rf anno_mage/static
cp -r web/frontend/dist anno_mage/static
echo "Building Python package..."
python -m build
echo "Done. Artifacts in dist/"
