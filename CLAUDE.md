# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Acta Machina (actamachina.com) is a Machine Learning blog. It uses Jekyll 4, Tailwind CSS (via `jekyll-postcss`), and deployed to GitHub Pages via GitHub Actions.

## Commands

```bash
# Install dependencies (first time or after Gemfile/package.json changes)
bundle install
npm install

# Local dev server with live reload
bundle exec jekyll serve

# Production build
JEKYLL_ENV=production bundle exec jekyll build

# Format templates and CSS
npx prettier --write .
```

There are no tests. Ruby 3.4.4, Node 22.17.0.

## Architecture

### Content Types

**Posts** (`_posts/YYYY-MM-DD-slug.md`) — long-form blog articles. Front matter: `layout: post`, `title`, `description`, `authors` (array of keys from `_data/authors.yml`), `x`/`y` (integer percentages for hero image crop position).

**Notes** (`_notes/1970-01-01-slug.md`) — reference cheatsheets using a multi-column masonry layout. Content is structured as `<section class="... break-inside-avoid-column ...">` blocks to prevent column breaks mid-section. Notes use `layout: note`.

### Layout Hierarchy

`base.html` → wraps all pages with `<html>`, `<head>`, sticky header, `<main>`. All other layouts extend `base`.

The `post` layout renders a `<article>` with Tailwind Typography prose classes. The `note` layout renders a `columns-sm` masonry grid. Both support dark mode via `dark:prose-invert` and `dark:bg-zinc-900`.

### CSS Pipeline

`assets/css/main.css` contains only Tailwind directives (`@tailwind base/components/utilities`) plus small SVG/MathJax overrides. `jekyll-postcss` runs Tailwind → Autoprefixer → cssnano (production only) on every build. Tailwind config scans `_includes`, `_layouts`, `_posts`, `_notes`, and root `*.md`/`*.html` for class names.

The `tailwind.config.js` includes a custom `prose-inline-code` variant and the `@tailwindcss/typography` plugin. The safelist includes `block hidden dark:block dark:hidden` (used by Mermaid diagrams).

### Mermaid Diagrams

`_plugins/mermaid_renderer.rb` is a Jekyll generator that pre-renders all ` ```mermaid ``` ` blocks to inline SVG before the site builds. It renders both a `neutral` (light) and `dark` theme SVG, caches them by SHA-256 hash in `.mermaid-cache/`, and injects both SVGs with `dark:hidden`/`dark:block` classes for CSS-driven theme switching. Requires `npx mmdc` (from `@mermaid-js/mermaid-cli`) to be available. Delete entries from `.mermaid-cache/` to force re-render.

### Front-end Libraries (CDN)

Loaded in `_includes/head.html` for all pages:
- **MathJax 3** — LaTeX math rendering
- **Plotly 2** — interactive charts
- **Tailwind Plus Elements** (`@tailwindplus/elements`) — provides `<el-dropdown>` / `<el-menu>` used in the site header

### Multi-site Header

The header dropdown (`_includes/header.html`) links to sibling sites (hyperfocal, Procedural) defined in `_config.yml` under `sites:`. Each site has a `font` and `url`. The active site's logo and font are rendered from this config. Site logos are SVG sprites referenced from `/assets/images/shared/logos.svg`.

### Authors

Authors are defined in `_data/authors.yml` (keys: `tlyleung`, etc). Reference them in post front matter as an array: `authors: [tlyleung]`.

### Deployment

GitHub Actions (`.github/workflows/main.yml`) builds on push to `main` and deploys to GitHub Pages. The workflow installs Ruby (via `ruby/setup-ruby` with bundler cache) and Node 22, runs `npm install`, then `bundle exec jekyll build` with `JEKYLL_ENV=production`.
