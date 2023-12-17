import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
const config: Config = {
    title: "Intro to Machine Learning",
    tagline: "6.390 IntroML Notes",
    favicon: "img/favicon.ico",

    // Set the production url of your site here
    url: "https://introml.mit.edu",
    // Set the /<baseUrl>/ pathname under which your site is served
    // For GitHub pages deployment, it is often '/<projectName>/'
    baseUrl: "/",

    // GitHub pages deployment config.
    // If you aren't using GitHub pages, you don't need these.
    organizationName: "shensquared", // Usually your GitHub org/user name.
    projectName: "introml-notes", // Usually your repo name.

    onBrokenLinks: "warn",
    onBrokenMarkdownLinks: "warn",
    plugins: [require.resolve("docusaurus-lunr-search")],
    // Even if you don't use internationalization, you can use this field to set
    // useful metadata like html lang. For example, if your site is Chinese, you
    // may want to replace "en" with "zh-Hans".
    i18n: {
        defaultLocale: "en",
        locales: ["en"],
    },

    presets: [
        [
            "classic",
            {
                docs: {
                    path: "notes",
                    remarkPlugins: [remarkMath],
                    rehypePlugins: [rehypeKatex],
                    sidebarPath: "./sidebars.ts",
                    routeBasePath: "notes",
                    editUrl: "https://github.com/shensquared/introml-notes",
                },
                blog: false,
                theme: {
                    customCss: "./src/css/custom.css",
                },
            } satisfies Preset.Options,
        ],
    ],
    stylesheets: [
        {
            href: "https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css",
            type: "text/css",
            integrity:
                "sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM",
            crossorigin: "anonymous",
        },
    ],
    themeConfig: {
        // Replace with your project's social card
        image: "img/390.png",
        navbar: {
            title: "IntroML",
            logo: {
                alt: "introml_logo",
                src: "img/390.png",
            },
            items: [
                {
                    type: "docSidebar",
                    sidebarId: "Sidebar",
                    position: "left",
                    label: "Notes",
                },
                {
                    href: "https://github.com/shensquared/introml-notes",
                    label: "GitHub",
                    position: "right",
                },
            ],
        },
        sidebar: {
            hideable: true,
            autoCollapseCategories: false,
        },
        prism: {
            theme: prismThemes.github,
            darkTheme: prismThemes.dracula,
        },
    } satisfies Preset.ThemeConfig,
};

export default config;
