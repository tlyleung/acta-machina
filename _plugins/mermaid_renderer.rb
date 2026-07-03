require 'fileutils'
require 'digest'
require 'open3'
require 'securerandom'
require 'tmpdir'

module Jekyll
  class MermaidRenderer < Jekyll::Generator
    safe true

    CACHE_DIR = File.join(Dir.pwd, '.mermaid-cache')

    def generate(site)
      FileUtils.mkdir_p(CACHE_DIR)
      site.collections['notes'].docs.each { |note| process_page(note) }
      site.collections['posts'].docs.each { |post| process_page(post) }
      site.pages.each { |page| process_page(page) }
    end

    def process_page(page)
      content = page.content
      updated_content = content.gsub(/```mermaid(.*?)```/m) do |match|
        mermaid_code = $1.strip

        # Generate a hash of the Mermaid code to determine if it has changed
        hash = Digest::SHA256.hexdigest(mermaid_code)
        cached_light_path = File.join(CACHE_DIR, "#{hash}_light.svg")
        cached_dark_path = File.join(CACHE_DIR, "#{hash}_dark.svg")

        light_svg = cached_light_path if File.exist?(cached_light_path)
        dark_svg = cached_dark_path if File.exist?(cached_dark_path)

        unless light_svg && dark_svg
          Jekyll.logger.info "Regenerating Mermaid Diagram", "Hash: #{hash}"

          # Render light and dark versions
          light_svg_content = render_mermaid_to_svg(mermaid_code, 'neutral')
          dark_svg_content = render_mermaid_to_svg(mermaid_code, 'dark')

          if light_svg_content && dark_svg_content
            File.write(cached_light_path, light_svg_content)
            File.write(cached_dark_path, dark_svg_content)

            light_svg = cached_light_path
            dark_svg = cached_dark_path
          else
            Jekyll.logger.error "Mermaid Render Failed", "Could not render one or both themes for Mermaid diagram."
            next match # Skip rendering and return the original block
          end
        end

        # Edge-label backgrounds must match whatever the diagram sits on: the
        # tinted note-section card, or the plain page body (posts and pages).
        on_card = page.data['layout'] == 'note'
        light_bg = on_card ? '#f3f3f3' : '#ffffff' # bg-zinc-950/5 over white, else body white
        dark_bg  = on_card ? '#242426' : '#18181b' # bg-white/5 over zinc-900, else body zinc-900

        # Combine the light and dark versions with proper class attributes
        light_svg_content = blend_edge_labels(File.read(light_svg), light_bg).gsub(/class="flowchart"/, 'class="flowchart light block dark:hidden"')
        dark_svg_content = blend_edge_labels(File.read(dark_svg), dark_bg).gsub(/class="flowchart"/, 'class="flowchart dark hidden dark:block"')

        <<~HTML
          #{light_svg_content}
          #{dark_svg_content}
        HTML
      end
    
      page.content = updated_content
    end

    # Recolour edge-label backgrounds to match whatever the diagram sits on, so
    # the label box masks the edge line behind it yet stays invisible against the
    # background. `bg` is that background's opaque colour, chosen at the call site
    # (the note-card composite, or the plain page body).
    def blend_edge_labels(svg, bg)
      svg
        .gsub(/(\.edgeLabel\s*\{[^}]*?background-color:\s*)[^;}]+/, "\\1#{bg}")
        .gsub(/(\.edgeLabel p\s*\{[^}]*?background-color:\s*)[^;}]+/, "\\1#{bg}")
        .gsub(/(\.labelBkg\s*\{[^}]*?background-color:\s*)[^;}]+/, "\\1#{bg}")
        .gsub(/(\.edgeLabel rect\s*\{[^}]*?)fill:\s*[^;}]+/, "\\1fill:#{bg}")
    end

    def render_mermaid_to_svg(mermaid_code, theme)
      Dir.mktmpdir do |dir|
        input_path = File.join(dir, 'diagram.mmd')
        output_path = File.join(dir, 'diagram.svg')

        # Generate a unique ID for each diagram to replace default "my-svg"
        svg_id = "mermaid-#{SecureRandom.hex(4)}"

        File.write(input_path, mermaid_code)

        css_path = File.expand_path('assets/css/mermaid.css', Dir.pwd)
        command = "npx mmdc -i #{input_path} -o #{output_path} --svgId #{svg_id} --theme #{theme} --cssFile #{css_path} --backgroundColor transparent --quiet"
        stdout, stderr, status = Open3.capture3(command)

        if status.success?
          Jekyll.logger.info "Mermaid SVG Generated", output_path
          File.read(output_path)
        else
          Jekyll.logger.error "Mermaid Render Error", stderr
          nil
        end
      end
    end
  end
end
