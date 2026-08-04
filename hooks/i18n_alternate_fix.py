"""
MkDocs hook to fix language switching links on the root index page (index.md).

By default, mkdocs-static-i18n skips updating config.extra["alternate"] links for
the root homepage (when page.url == "" or PurePath(page.url) == PurePath(".")),
leaving absolute domain-root links ("/zh/") instead of relative links ("zh/").
When the documentation is hosted in a subpath (or viewed locally), clicking "/zh/"
replaces the entire URL path instead of appending "zh" to the current site path.

In addition, MkDocs' AttributeDict can maintain separate object references for
instance attributes (config.extra.alternate) and dict items (config.extra["alternate"])
when attributes are assigned directly.

This hook runs at event priority -100 (after mkdocs-static-i18n at priority 50) and
ensures that relative alternate file URLs are used consistently for all pages across
both attribute and dict item references, including index.md.
"""
from mkdocs.plugins import event_priority


@event_priority(-100)
def on_page_context(context, page, config, nav, **kwargs):
    if hasattr(page, "file") and hasattr(page.file, "alternates"):
        for cfg in (config, context.get("config", config)):
            if cfg and hasattr(cfg, "extra"):
                alt_lists = []
                if hasattr(cfg.extra, "alternate") and isinstance(
                    cfg.extra.alternate, list
                ):
                    alt_lists.append(cfg.extra.alternate)
                if (
                    "alternate" in cfg.extra
                    and isinstance(cfg.extra["alternate"], list)
                    and cfg.extra["alternate"] not in alt_lists
                ):
                    alt_lists.append(cfg.extra["alternate"])
                for alt_list in alt_lists:
                    for alt in alt_list:
                        alt_lang = alt.get("lang")
                        if alt_lang in page.file.alternates:
                            alt["link"] = page.file.alternates[alt_lang].url
    return context
