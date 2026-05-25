---
name: Feature request
about: Suggest an idea for Skein
title: ""
labels: enhancement
---

## The problem

<!-- What's the friction you're hitting today? Concrete examples > abstract wishes. -->

## The proposal

<!-- What would you like Skein to do? CLI behavior, MCP tool, daemon
automation — be specific. -->

## Alternatives you've tried

<!-- workarounds, other tools, etc. -->

## Vision check

Skein's design constraint is "fewer commands doing more work
automatically; never grow the user toolkit." MCP tools the LLM invokes
are fine; new user-facing CLI commands need to clear a high bar. Does
your proposal fit?

- [ ] This is an MCP tool an agent would call (fits)
- [ ] This makes the daemon do something automatic that I currently do manually (fits)
- [ ] This folds into an existing command (`skein doctor`, `skein status`, etc.) (fits)
- [ ] This is a new top-level CLI command (justify why it can't be one of the above)
