#!/usr/bin/env python3
"""Test update_marketplace function."""

import json
from pathlib import Path
from sync_types import ExternalSource, Skill, SyncResult

# Create a temporary marketplace for testing
temp_marketplace = Path("/tmp/test_marketplace.json")
test_data = {
    "$schema": "https://anthropic.com/claude-code/marketplace.schema.json",
    "name": "awesome-ascend-skills",
    "version": "1.0.0",
    "plugins": [
        {
            "name": "npu-smi",
            "description": "Huawei Ascend NPU npu-smi command reference.",
            "source": "./npu-smi",
            "category": "operations",
        }
    ],
}

with open(temp_marketplace, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)

# Create test data
source = ExternalSource(
    name="test-repo", url="https://github.com/test/repo", branch="main"
)
skills = [
    Skill(
        name="test-skill-1",
        path=Path("/tmp/external/test-repo/test-skill-1"),
        source=source,
        has_skill_md=True,
    ),
    Skill(
        name="test-skill-2",
        path=Path("/tmp/external/test-repo/test-skill-2"),
        source=source,
        has_skill_md=True,
    ),
]
results = SyncResult(synced=["test-skill-1", "test-skill-2"], skipped=[], errors=[])

# Import and run the function
from sync_external_skills import update_marketplace

# Run the function
update_marketplace(skills, results)

# Verify the result
with open(temp_marketplace, "r", encoding="utf-8") as f:
    result_data = json.load(f)

# Check that external skills were added
external_plugins = [
    p for p in result_data.get("plugins", []) if p.get("external") == True
]
print(f"✅ Found {len(external_plugins)} external plugins in marketplace")

# Verify each external skill was added
expected_names = ["external-test-repo-test-skill-1", "external-test-repo-test-skill-2"]
for expected_name in expected_names:
    if any(p["name"] == expected_name for p in external_plugins):
        print(f"✅ {expected_name} added correctly")
    else:
        print(f"❌ {expected_name} NOT found in marketplace")

# Verify source and category are correct
if len(external_plugins) == 2:
    p1 = [
        p for p in external_plugins if p["name"] == "external-test-repo-test-skill-1"
    ][0]
    assert p1["source"] == "./external/test-repo/test-skill-1"
    assert p1["category"] == "external"
    assert p1["external"] == True
    assert p1["source-url"] == "https://github.com/test/repo"
    print("✅ External skill properties verified")
else:
    print(f"❌ Expected 2 external plugins, got {len(external_plugins)}")

# Verify existing plugin is not modified
original_plugin = [p for p in result_data.get("plugins", []) if p["name"] == "npu-smi"][
    0
]
assert original_plugin["name"] == "npu-smi"
assert original_plugin["category"] == "operations"
print("✅ Existing plugin not modified")

print("\n✅ All tests passed!")

# Cleanup
temp_marketplace.unlink()
