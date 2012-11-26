(require 'org-publish)
(require 'org-exp-bibtex)
(setq org-publish-project-alist
      '(

	("CompNeuro"
	 :base-directory "org/"
	 :base-extension "org"
	 :publishing-directory "html/"
	 :Publishing-function org-publish-org-to-html
	 :recursive t
	 :section-numbers nil
	 :html-postamble "<hr />%a | %d<br>This <span xmlns:dct=\"http://purl.org/dc/terms/\" href=\"http://purl.org/dc/dcmitype/Text\" rel=\"dct:type\">work</span> is licensed under a <a rel=\"license\" href=\"http://creativecommons.org/licenses/by-nc-sa/3.0/\">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a><br><a rel=\"license\" href=\"http://creativecommons.org/licenses/by-nc-sa/3.0/\"><img alt=\"Creative Commons License\" style=\"border-width:0\" src=\"http://i.creativecommons.org/l/by-nc-sa/3.0/80x15.png\" /></a><br />"
	 :language en
	 :link-home "index.html"
	 :link-up "index.html"
	 :style "<link rel=\"stylesheet\" type=\"text/css\" href=\"mystyle.css\" />"
	 )

	("CompNeuro_html"
	 :base-directory "html/"
	 :base-extension "css\\|html"
	 :publishing-directory "/ssh:admin@leviathan.ssc.uwo.ca:~/gribblelab.org/compneuro/"
	 :publishing-function org-publish-attachment
	 :recursive t
	 )

	("CompNeuro_bibhtml"
	 :base-directory "org/"
	 :base-extension "html\\|css"
	 :publishing-directory "/ssh:admin@leviathan.ssc.uwo.ca:~/gribblelab.org/compneuro/"
	 :publishing-function org-publish-attachment
	 :recursive t
	 )

	("CompNeuro_figs"
	 :base-directory "figs/"
	 :base-extension "png\\|jpg\\|pdf"
	 :publishing-directory "/ssh:admin@leviathan.ssc.uwo.ca:~/gribblelab.org/compneuro/figs/"
	 :publishing-function org-publish-attachment
	 :recursive t
	 )

	("CompNeuro_code"
	 :base-directory "code/"
	 :base-extension "c\\|h\\|txt\\|csv\\|py\\|tgz\\|pickle\\|m"
	 :publishing-directory "/ssh:admin@leviathan.ssc.uwo.ca:~/gribblelab.org/compneuro/code/"
	 :publishing-function org-publish-attachment
	 :recursive t
	 )

	("org" :components ("CompNeuro" "CompNeuro_html" "CompNeuro_bibhtml" "CompNeuro_figs" "CompNeuro_code"))))

(org-publish-project "CompNeuro")
(org-publish-project "CompNeuro_html")
(org-publish-project "CompNeuro_bibhtml")
(org-publish-project "CompNeuro_figs")
(org-publish-project "CompNeuro_code")
